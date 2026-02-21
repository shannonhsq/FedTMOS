


import numpy as np 



class Client:
    def __init__(self, train_x, train_y, test_x, test_y, model, numclasses=10):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model = model
        self.numclasses = numclasses
  
 
    def train(self, model_num, epochs = 1):
        self.model[model_num].fit(self.train_x[model_num],self.train_y,epochs = epochs, incremental = True)
        
    
    def test(self, data_x , data_y,  return_probs = False):
        votes = np.zeros((len(data_x[0]), self.numclasses), dtype= np.float32)
        all_conf = []
        for i in range(len(self.model)):
            
            accY, all_cc = self.model[i].get_score(data_x[i]) 
            all_conf.append(all_cc)
        for j in range(len(data_x)):
        
            vote = 1.0*all_votes[i]/(np.max(all_votes) - np.min(all_votes)) 
            votes[j] += vote 

        prediction = votes.argmax(axis = 1)
        acc_test = (100*(prediction== data_y).mean())
        
        if return_probs:
            return prediction, acc_test, all_conf
        else:
            return prediction, acc_test
    
    def handle_uploading(self, model_num):
        print('uploading....')
        unique_labels = np.unique(self.train_y)
        
        params = self.model[model_num].get_state()
        num_clauses = self.model[model_num].__dict__['number_of_clauses']
        num_states = len(params[0][1])
        ta_chunk = self.model[model_num].__dict__['number_of_ta_chunks']
        num_state_bits = self.model[model_num].__dict__['number_of_state_bits']
        clause_weights  = np.zeros(len(params[0][0]), dtype = 'uint32')
        
        
        ta_state =  np.zeros(len(params[0][1]), dtype = 'uint32')
        new_parameters =  [[np.copy(clause_weights), np.copy(ta_state)] for _ in range(self.numclasses)]

        for i in range(self.numclasses):
            if i in unique_labels:
                ta_state = params[i][1]
                clause_weights = params[i][0]
            else:
                ta_state =  np.zeros(len(params[0][1]), dtype = 'uint32')
                clause_weights =  np.zeros(len(params[0][0]), dtype = 'uint32')
            
            pos = 0
            sig_bit_num =0 
            for j in range(num_clauses):
                for k in range(ta_chunk):
                        for b in range(num_state_bits-1):
                            ta_state[pos] = 0
                            pos +=1
                        pos+=1
                        sig_bit_num +=1
                
            new_parameters[i] = (clause_weights, ta_state)
            

        self.model[model_num].set_state(new_parameters)
        return self.model[model_num]

    def handle_non_iid(self, model_num): 
        print('handle non iid')
    

        unique_labels = np.unique(self.train_y)
    

        params = self.model[model_num].get_state()
    
        clause_weights  = np.zeros(len(params[0][0]), dtype = 'uint32')
        ta_state = np.zeros(len(params[0][1]), dtype = 'uint32')

    
        new_parameters =  [[np.copy(clause_weights), np.copy(ta_state)] for _ in range(self.numclasses)]
    
        for i in range(self.numclasses):
            clause_weights  = np.zeros(self.model[model_num].number_of_clauses, dtype = 'uint32')
            if i in unique_labels:
                new_parameters[i] = params[i]


        self.model[model_num].set_state(new_parameters)
        return self.model[model_num]





    
     

   

