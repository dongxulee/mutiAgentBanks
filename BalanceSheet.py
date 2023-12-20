import pandas as pd

class BalanceSheet:
    def __init__(self, portfolio=0, lending=0, borrowing=0, deposit=0):
        # assets
        self._portfolio = portfolio       
        self._lending = lending         
        # liabilities
        self._borrowing = borrowing       
        self._deposit = deposit
    
    @property
    def portfolio(self):
        return self._portfolio
    
    @portfolio.setter
    def portfolio(self, value):
        if value < 0:
            raise ValueError("Portfolio cannot be negative")
        else:   
            self._portfolio = value
        
    @property
    def lending(self):
        return self._lending
    
    @lending.setter
    def lending(self, value):
        if value < 0:
            raise ValueError("Lending cannot be negative")
        else:   
            self._lending = value

    @property
    def borrowing(self):
        return self._borrowing
    
    @borrowing.setter
    def borrowing(self, value):
        if value < 0:
            raise ValueError("Borrowing cannot be negative")
        else:   
            self._borrowing = value
        
    @property
    def deposit(self):
        return self._deposit
    
    @deposit.setter
    def deposit(self, value):
        if value < 0:
            raise ValueError("Deposit cannot be negative")
        else:   
            self._deposit = value
            
    @property
    def equity(self):
        return self._portfolio + self._lending - self._borrowing - self._deposit
    
    @property
    def leverage(self):
        return (self._portfolio + self._lending) / self.equity
    
    @property
    def default(self):
        return self.equity <= 0
    
    def print(self):
        return display(pd.DataFrame(data = [[self._portfolio, self._lending, self._borrowing, self._deposit, self.equity, self.leverage, self.default]], columns = ['portfolio', 'lending', 'borrowing', 'deposit', 'equity', 'leverage', 'default']))
    
    
# test functions
if __name__ == '__main__':
    b = BalanceSheet(10,0,0,8)
    b.lending = 2
    b.borrowing = 1
    b.print()
    b.portfolio = 11
    b.print()
    b.portfolio = 5
    b.print()
    b.deposit = 10
    b.print()