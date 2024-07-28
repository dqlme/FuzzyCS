


class SelectionBase:
    """
    Base class for selection methods.
    """
    
    def __init__(self, args):
        self.args = args
        
    def select(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Selects k individuals from the population.
        
        Parameters:
        k (int): The number of individuals to select.
        
        Returns:

        """
                
        pass