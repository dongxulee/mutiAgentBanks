import numpy as np
import random
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiEnvDict
from BalanceSheet import BalanceSheet
from eisenbergNoe import eisenbergNoe
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

class BankingSystemEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        self.t = 0  # Time step
        self.num_banks = 5  # Number of banks in the system
        self.num_borrowing = 1
        # Define the structure of a bank's balance sheet
        # For simplicity, let's assume each balance sheet consists of:
        # [portfolio, lending, borrowing, deposit, equity, leverage, default]
        self.balance_sheet_size = 7

        self._agent_ids = set()
        # Initialize balance sheets for each bank
        self.balance_sheets = {
            0: BalanceSheet(3, 0, 0, 0.8*3),
            1: BalanceSheet(2, 0, 0, 0.8*2),
            2: BalanceSheet(1, 0, 0, 0.8*1),
            3: BalanceSheet(0.3, 0, 0, 0.8*0.3),
            4: BalanceSheet(0.5, 0, 0, 0.8*0.5)
        }
        
        # high level parameters
        self.alpha = 0.5
        self.beta = 0.9
        fedRate = 0.04
        portfolioReturnRate = 0.1
        returnVolatiliy = 0.18
        returnCorrelation = 0.5
        concentrationParameter = None
        # interest rate  
        self.depositReserve = 0.2
        self.fedRate = (fedRate+1)**(1/252) - 1
        # portfolio return rate
        self.portfolioReturnRate = (portfolioReturnRate+1)**(1/252) - 1
        # portfolio return volatility
        self.returnVolatiliy = returnVolatiliy/np.sqrt(252)
        # return correlation matrix
        cMatrix = np.ones((self.num_banks,self.num_banks))*returnCorrelation
        np.fill_diagonal(cMatrix, 1)
        self.Cholesky = np.linalg.cholesky(cMatrix * self.returnVolatiliy**2)
        # liability matrix 
        self.L = np.zeros((self.num_banks,self.num_banks))
        # asset matrix
        self.e = np.array([self.balance_sheets[i].portfolio for i in range(self.num_banks)]).reshape(-1,1)
        # deposit matrix
        self.d = np.array([self.balance_sheets[i].deposit for i in range(self.num_banks)]).reshape(-1,1)
        # define concentration parameter
        if concentrationParameter is None:
            self.concentrationParameter = np.ones((self.num_banks,self.num_banks))
            np.fill_diagonal(self.concentrationParameter, 0.)
            self.trustMatrix = self.concentrationParameter / (self.num_banks - 1)
        else:
            self.concentrationParameter = concentrationParameter
            self.trustMatrix = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True)

        # Define the action space for each bank:
        # Action could be a vector [borrow_amount, lend_amount]
        # For simplicity, let's assume each bank can borrow a percentage of its equity
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        # Define the observation space for each bank:
        # Observation is the agent's balance sheet and borrowing target (another bank's balance sheet)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(2, self.balance_sheet_size),
            dtype=np.float32
        )
        
    def clearingDebt(self): 
        # Returns the new portfolio value after clearing debt
        _, e = eisenbergNoe(self.L*(1+self.fedRate), self.e, self.alpha, self.beta)
        self.e = e
        insolventBanks = np.where(self.e - self.d <= 0)[0]
        # reset the Liabilities matrix after clearing debt
        self.L = np.zeros((self.num_banks,self.num_banks))
        # reset the borrowing and lending matrix after clearing debt
        for i in range(self.num_banks):
            self.balance_sheets[i].borrowing = 0
            self.balance_sheets[i].lending = 0
        if len(insolventBanks) > 0:
            self.concentrationParameter[:,insolventBanks] = 0

    def returnOnPortfolio(self):
        # Return on the portfolio:
        self.e += (self.e - self.d*self.depositReserve) * (self.portfolioReturnRate + (self.Cholesky @ np.random.randn(self.num_banks,1)))
        
    def updateTrustMatrix(self):
        # add time decay of concentration parameter
        self.concentrationParameter = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True) * (self.num_banks - 1) * self.num_borrowing
        self.trustMatrix = self.concentrationParameter / (self.num_banks - 1) / self.num_borrowing
        

    def step(self, action_dict):
        # Initialize the return values
        obs, rewards, done, info = {}, {}, {}, {}
        self.t += 1
        # Process actions for each bank, borrowing and lending
        for bank_id, action in action_dict.items():
            if self.balance_sheets[bank_id].default == True:
                continue
            # Randomly select another bank to request a loan
            target_bank = np.random.choice(self.num_banks, p=self.trustMatrix[bank_id])

            # Get the observation of it's own balance sheet and the target bank's balance sheet
            obs[bank_id] = self._get_observation(bank_id, target_bank)
            info[bank_id] = {"target_bank": target_bank}
            # Amount the bank wants to borrow
            borrow_amount = action[0]*self.balance_sheets[bank_id].equity  # Borrow a percentage of equity

            # Check if the target bank decides to lend based on balance sheets
            if self._decide_to_lend(bank_id, target_bank, borrow_amount):
                # Update balance sheets for borrowing and lending
                self._process_transaction(bank_id, target_bank, borrow_amount)

        self.e = np.array([self.balance_sheets[i].portfolio for i in range(self.num_banks)]).reshape(-1,1)
        self.returnOnPortfolio()
        self.clearingDebt()
        self.updateTrustMatrix()
        # calculate rewards for each bank and set the new portfolio value
        for bank_id in range(self.num_banks):
            rewards[bank_id] = self._calculate_reward(bank_id)
            self.balance_sheets[bank_id].portfolio = self.e[bank_id]

        # Check if the environment is done
        done["__all__"] = self._is_done()  # Define this method

        return obs, rewards, done, info

    def _decide_to_lend(self, borrower_id, lender_id, amount):
        # Implement logic to decide if lender should lend money based on balance sheets
        if self.balance_sheets[borrower_id].leverage < 10 and self.balance_sheets[lender_id].portfolio > amount:
            return True
        else:
            return False

    def _process_transaction(self, borrower_id, lender_id, amount):
        # Update balance sheets for both borrower and lender
        self.balance_sheets[borrower_id].portfolio += amount  # Increase borrower's cash
        self.balance_sheets[borrower_id].borrowing += amount  # Increase borrower's borrowing
        self.balance_sheets[lender_id].portfolio -= amount    # Decrease lender's cash
        self.balance_sheets[lender_id].lending += amount      # Increase lender's lending
        self.L[borrower_id, lender_id] += amount
        self.concentrationParameter[borrower_id, lender_id] += 1.
        
    def _get_observation(self, bank_id, target_bank=None):
        if target_bank == None:
            return np.array([self.balance_sheets[bank_id].values, np.zeros(self.balance_sheet_size)])
        else:
            return np.array([self.balance_sheets[bank_id].values, self.balance_sheets[target_bank].values])

    def _calculate_reward(self, bank_id):
        if self.balance_sheets[bank_id].default == True:
            return -1
        else:
            # return the reward as the change in equity
            return (self.e[bank_id] - self.balance_sheets[bank_id].portfolio) / self.balance_sheets[bank_id].equity
    
    def _is_done(self):
        # Define your own termination criteria
        if self.t >= 500:
            return True
        else:
            return False
    
    def seed(self, seed=None):
        # Set the seed for this env's random number generator(s).
        # If your environment does not use any RNGs, you can leave this method empty.
        random.seed(seed)
        np.random.seed(seed)
        
    def reset(self, *, seed=None, options=None):
        self._agent_ids.clear()
        for i in range(self.num_banks):
            self._agent_ids.add(i)
        
        # Reset balance sheets to initial state
        self.balance_sheets = {
            0: BalanceSheet(3, 0, 0, 0.8*3),
            1: BalanceSheet(2, 0, 0, 0.8*2),
            2: BalanceSheet(1, 0, 0, 0.8*1),
            3: BalanceSheet(0.3, 0, 0, 0.8*0.3),
            4: BalanceSheet(0.5, 0, 0, 0.8*0.5)
        }
        # Optionally, store previous balance sh
        # Initialize observations for each bank
        obs = {
            bank_id: self._get_observation(bank_id) for bank_id in range(self.num_banks)
        }
        info = {}
        return obs, info

if __name__ == '__main__':
    import ray
    from ray.rllib.algorithms import ppo
    ray.init()    
    algo = ppo.PPO(env=BankingSystemEnv, config={
        "env_config": {},  # config to pass to env class
    })
    while True:
        print(algo.train())