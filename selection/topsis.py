from .client_selection import ClientSelection
import torch
import numpy as np


class Topsis(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)
        self.epochs = 1000
    # def init(self, global_m, local_models):
    #     self.prev_global_m = global_m
    #     self.gradients = self.get_gradients(global_m, local_models)    
        
    def setup(self, n_samples):
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        self.weights = n_samples / np.sum(n_samples)


    def select(self, n, client_idxs, local_models, global_model, metric, running_time, comm_time, gr):
        '''
        因素：数据规模、梯度、损失、通信代价、计算资源
        '''
        # 1.数据规模
        weights = np.take(self.weights, client_idxs)
        # 2.梯度
        # gradNorm = []
        # for local_model in local_models:
        #     # local_grad = local_model.linear_2.weight.grad.data #head.conv.weight.grad.data
            
        #     local_grad = list(local_model.state_dict().values())[-2].grad.data
        #     local_grad_norm = torch.sum(torch.abs(local_grad)).cpu().numpy()
            # gradNorm.append(local_grad_norm)
        gradients = self.get_gradients(global_model, local_models) 
        # gradients = gradients.cpu().numpy()
        gradientNorm = [np.linalg.norm(gradients[i][-2]) for i in range(100)]
        # import code 
        # code.interact(local=dict(globals(), **locals()))
        # print(gradients.shape)
        #  [np.linalg.norm(gradients[i][-2]) for i in range(100)] 
        gradNorm = gradientNorm / np.sum(gradientNorm)
        # 3.损失
        losses = metric/np.linalg.norm(metric)
        
        # 4.通信代价、计算资源、计算时间
        running_time = running_time/np.linalg.norm(running_time) # 归一化处理，避免数值问题
        comm_time = comm_time/np.linalg.norm(comm_time) 
        
        # 5.计算总得分，并选择n个客户端
        best_v = [np.max(weights),np.max(gradNorm),np.max(losses),np.min(running_time),np.min(comm_time)] # 取最大值作为权重，避免负数影响
        worst_v = [np.min(weights),np.min(gradNorm),np.min(losses),np.max(running_time),np.max(comm_time)] # 取最小值作为权重，避免负数影响
        # dist_2_bestv = np.sqrt(np.sum((weights - best_v[0])**2)) + (gradNorm - best_v[1])**2 + (losses - best_v[2])**2 + (running_time - best_v[3])**2 + (comm_time - best_v[4])**2
        dist_2_bestv = [np.sqrt((weights[i]-best_v[0])**2+(gradNorm[i]-best_v[1])**2+(losses[i]-best_v[2])**2 \
                                +(running_time[i]-best_v[3])**2+(comm_time[i]-best_v[4])**2) for i in client_idxs]
        dist_2_worstv = [np.sqrt((weights[i]-worst_v[0])**2+(gradNorm[i]-worst_v[1])**2+(losses[i]-worst_v[2])**2 \
                                + (running_time[i]-worst_v[3])**2+(comm_time[i]-worst_v[4])**2) for i in client_idxs]           
            
            
        dist = np.array(dist_2_worstv) / (np.array(dist_2_bestv) + np.array(dist_2_worstv))# 相对贴近度
        # select
        if gr < int(0.1 * self.epochs):
            selected_client_idxs = np.argsort(dist)[:n]
        elif gr>int(0.1 * self.epochs) & gr<int(0.3* self.epochs):  
            nidx = np.random.choice(np.arange(10,90),n)
            selected_client_idxs = np.argsort(dist)[nidx]
        else:
            selected_client_idxs = np.argsort(dist)[-n:]

        return selected_client_idxs.astype(int)
    
    def get_gradients(self, global_m, local_models):
        """
        return the `representative gradient` formed by the difference
        between the local work and the sent global model
        """
        global_m = global_m.to('cpu')
        local_model_params = []
        for model in local_models:
            local_model_params += [[tens.detach() for tens in list(model.parameters())[-2]]] #.numpy()

        global_model_params = [tens.detach() for tens in list(global_m.parameters())[-2]]

        local_model_grads = []
        for local_params in local_model_params:
            local_model_grads += [[local_weights - global_weights
                                   for local_weights, global_weights in
                                   zip(local_params, global_model_params)]]
        del local_model_params
        del global_model_params
        torch.cuda.empty_cache()
        return local_model_grads







