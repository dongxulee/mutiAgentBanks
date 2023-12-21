import numpy as np 

def eisenbergNoe(L, e, alpha, beta):
    '''
        eisenbergNoe algorithm with recovery rate alpha and beta
        
        Input:
            L: liability matrix, (N,N)
            e: net asset, (N,1)
            alpha: recovery rate on asset 
            beta: recovery rate on interbank asset 
        Output:
            A: payments to creditors
            e_new: net asset values 
            insolventBanks: indices of insolvent banks
    '''
    N = e.size 
    # asset matrices shapes 
    # assert(L.shape == (N,N))
    # assert(e.shape == (N,1))
    # assert(alpha >= 0 and alpha <= 1)
    # assert(beta >= 0 and beta <= 1)
    # total liabilities of each bank
    L_bar = L.sum(axis = 1, keepdims=True)
    # relative liability matrix, if no obligations, set to 0
    noneZeroIndex = np.where(L_bar != 0)[0]
    Pi = np.zeros((N,N))
    Pi[noneZeroIndex] = L[noneZeroIndex]/L_bar[noneZeroIndex]
    # initial guess
    A = L_bar.copy()
    insolventBanks = np.array([])
    solventBanks = np.array([])
    while(True):         
        V = Pi.T @ A + e - L_bar
        insolventBanks_current = np.where(V < 0)[0]
        solventBanks_current = np.where(V >= 0)[0]
        if set(insolventBanks) == set(insolventBanks_current):
            break
        else:
            insolventBanks = insolventBanks_current.copy()
            solventBanks = solventBanks_current.copy()
            # update solvent banks payments 
            A[solventBanks] = L_bar[solventBanks]
            # construct and solve the linear system to get x
            x = np.linalg.inv(np.eye(insolventBanks.size)- beta*(Pi[insolventBanks].T)[insolventBanks]) @ (alpha * e[insolventBanks] + beta*((Pi[solventBanks].T)[insolventBanks]@L_bar[solventBanks]))
            # update insolvent banks payments
            A[insolventBanks] = x
    return A, (Pi.T @ A + e - A)*(A >= L_bar)

# function to test if A is the fixed point of phi
def phi(A, L, e, alpha, beta):
    N = e.size 
    L_bar = L.sum(axis = 1, keepdims=True)
    # relative liability matrix, if no obligations, set to 0
    noneZeroIndex = np.where(L_bar != 0)[0]
    Pi = np.zeros((N,N))
    Pi[noneZeroIndex] = L[noneZeroIndex]/L_bar[noneZeroIndex]
    A_prime = np.zeros(A.shape)
    solvent = np.where(L_bar <= e + Pi.T @ A)[0]
    insolvent = np.where(L_bar > e + Pi.T @ A)[0]
    A_prime[solvent] = L_bar[solvent]
    A_prime[insolvent] = alpha*e[insolvent] + beta*Pi.T[insolvent] @ A 
    return A_prime



if __name__ == "__main__":
    # liability matrix 
    alpha = 0.5
    beta = 0.5
    L = np.array([[0,2.2],[2.2,0]])
    e = np.array([[1],[1]])
    A,e = eisenbergNoe(L, e, alpha, beta)
    A_prime = phi(A, L, e, alpha, beta)
    print("phi(L) == L?", np.isclose(A_prime,A).all())
