import numpy as np
import random
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from BalanceSheet import BalanceSheet

class BankingSystemEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__()
        self.t = 0  # Time step 
        self.num_banks = 5  # Number of banks in the system

        # Define the structure of a bank's balance sheet
        # For simplicity, let's assume each balance sheet consists of:
        # [portfolio, lending, borrowing, deposit, equity, leverage, default]
        self.balance_sheet_size = 7

        # Initialize balance sheets for each bank
        self.balance_sheets = {
            'bank_1': BalanceSheet(3, 0, 0, 0.8*3),
            'bank_2': BalanceSheet(2, 0, 0, 0.8*2),
            'bank_3': BalanceSheet(1, 0, 0, 0.8*1), 
            'bank_4': BalanceSheet(0.3, 0, 0, 0.8*0.3),
            'bank_5': BalanceSheet(0.5, 0, 0, 0.8*0.5),
        }

        # Define the action space for each bank:
        # Action could be a vector [borrow_amount, lend_amount]
        # For simplicity, let's assume each bank can borrow or lend a fixed amount
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),  # No borrowing, No lending
            high=np.array([50, 50]),  # Max borrowing, Max lending
            dtype=np.float32
        )

        # Define the observation space for each bank:
        # Observation is the entire system's balance sheets
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.num_banks * self.balance_sheet_size,),
            dtype=np.float32
        )
        

    def step(self, action_dict):
        # Initialize the return values
        obs, rewards, done, info = {}, {}, {}, {}

        # Process actions for each bank
        for bank_id, action in action_dict.items():
            # Randomly select another bank to request a loan
            other_banks = [b for b in self.balance_sheets.keys() if b != bank_id]
            target_bank = random.choice(other_banks)

            # Amount the bank wants to borrow
            borrow_amount = action[0]

            # Check if the target bank decides to lend based on balance sheets
            if self._decide_to_lend(bank_id, target_bank, borrow_amount):
                # Update balance sheets for borrowing and lending
                self._process_transaction(bank_id, target_bank, borrow_amount)

            # Construct observations, rewards, and info
            obs[bank_id] = self._get_observation(bank_id)
            rewards[bank_id] = self._calculate_reward(bank_id)  # Define this method
            info[bank_id] = {"target_bank": target_bank}

        # Check if the environment is done (you can define your own criteria)
        done["__all__"] = self._is_done()  # Define this method

        return obs, rewards, done, info

    def _decide_to_lend(self, borrower_id, lender_id, amount):
        # Implement logic to decide if lender should lend money based on balance sheets
        # For simplicity, let's say the lender lends if it has more cash than the borrower
        return self.balance_sheets[lender_id][0] > self.balance_sheets[borrower_id][0]

    def _process_transaction(self, borrower_id, lender_id, amount):
        # Update balance sheets for both borrower and lender
        self.balance_sheets[borrower_id][0] += amount  # Increase borrower's cash
        self.balance_sheets[lender_id][0] -= amount   # Decrease lender's cash

    def _get_observation(self, bank_id):
        # Construct observation (entire system's balance sheets)
        return np.concatenate([self.balance_sheets[b].flatten() for b in self.balance_sheets])

    def _is_done(self):
        # Define your own termination criteria
        return False
    


if __name__ == '__main__':
    env = BankingSystemEnv() 
    obs, infos = env.reset(seed=42, options={}) 
    print(obs) 