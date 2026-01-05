import copy
import random
import time
import torch
import numpy as np
from flcore.clients.clientHkd import clientHKD
from flcore.servers.serverbase import Server
from threading import Thread
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import os
import h5py


class FedHKD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientHKD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        # self.T_start = args.T_start
        # self.T_end = args.T_end
        # self.energy = self.T_start
        # self.compressed_param = {}

        # 确保这些属性在初始化时就被设置
        self.model_str = args.model_str if hasattr(args, 'model_str') else 'unknown_local'
        self.global_model_type = args.global_model_type if hasattr(args, 'global_model_type') else 'unknown_global'

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            self.current_round = i
            self.selected_clients = self.select_clients()
            
            # Pass current round number to clients
            for client_obj in self.selected_clients:
                client_obj.current_round = i 
            
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()
                # 保存结果到h5文件
                self.save_results()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            # self.decomposition()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            # self.energy = self.T_start + ((1 + i) / self.global_rounds) * (self.T_end - self.T_start)

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientHKD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def send_models(self):
        assert (len(self.clients) > 0)

        params_to_send = {}
        # 检查 self.global_model 的类型
        # 如果是PyTorch模型（通常在第一轮），则从state_dict()转换
        if isinstance(self.global_model, torch.nn.Module):
            for name, param in self.global_model.state_dict().items():
                params_to_send[name] = param.cpu().numpy()
        # 如果已经是参数字典（聚合后的结果），则直接使用
        else:
            params_to_send = self.global_model

        for client in self.clients:
            start_time = time.time()

            # Update client's current round before sending parameters
            if hasattr(client, 'current_round'):
                client.current_round = self.current_round if hasattr(self, 'current_round') else 0

            # energy 参数不再需要，可以传 None 或移除
            client.set_parameters(params_to_send, None) 

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)    
    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            
            # 传递全局模型信息
            client_args = copy.deepcopy(self.args)
            if hasattr(self.args, 'global_model'):
                client_args.global_model = self.args.global_model
                
            client = clientObj(client_args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_models = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                # recover
                # for k in client.compressed_param.keys():
                #     if len(client.compressed_param[k]) == 3:
                #         # use np.matmul to support high-dimensional CNN param
                #         client.compressed_param[k] = np.matmul(
                #             client.compressed_param[k][0] * client.compressed_param[k][1][..., None, :],
                #             client.compressed_param[k][2])

                self.uploaded_models.append(client.compressed_param)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for k in self.global_model.keys():
            self.global_model[k] = np.zeros_like(self.global_model[k])

        # use 1/len(self.uploaded_models) as the weight for privacy and fairness
        for client_model in self.uploaded_models:
            self.add_parameters(1 / len(self.uploaded_models), client_model)

    def add_parameters(self, w, client_model):
        for server_k, client_k in zip(self.global_model.keys(), client_model.keys()):
            self.global_model[server_k] += client_model[client_k] * w

    # Override train_metrics from serverbase.py to handle dicts from clientHkd.py 为输出三个损失值
    def train_metrics(self):
        all_metrics_dicts = []
        # self.clients should be a list of clientHKD objects selected for this round
        # If evaluate is called on all clients, it would be self.clients
        # If on selected_clients, then self.selected_clients
        # Assuming self.clients for broad metric collection, or self.selected_clients if that's the target set
        clients_to_evaluate = self.selected_clients if self.selected_clients else self.clients
    
        for c in clients_to_evaluate:
            # Ensure current_round is passed to client for warmup factor calculation in client.train_metrics
            if hasattr(c, 'current_round') and hasattr(self, 'current_round'):
                c.current_round = self.current_round
            else:
                # Fallback or warning if current_round is not available, though it should be for HKD
                if hasattr(c, 'current_round'): # Client expects it
                    print(f"Warning: Server current_round not set, but client {c.id} expects it. Defaulting client current_round to 0.")
                    c.current_round = 0
    
            metrics_dict = c.train_metrics()  # clientHkd.train_metrics() returns a dictionary
            all_metrics_dicts.append(metrics_dict)
        return all_metrics_dicts

    def evaluate(self, acc=None, loss=None):
        # stats will be: (list_correct_predictions, list_sum_test_loss, list_num_samples, list_auc)
        stats = self.test_metrics()
        stats_train = self.train_metrics()
    
        # Unpack test statistics
        list_correct_predictions = stats[0]
        list_sum_test_loss = stats[1]
        list_num_samples = stats[2]
        # list_auc = stats[3] # AUC is not directly used for overall test_acc/loss reporting here
    
        total_test_correct = sum(list_correct_predictions)
        total_test_loss_sum = sum(list_sum_test_loss)
        total_test_samples = sum(list_num_samples)
    
        if total_test_samples == 0:
            print("Warning: No test samples were processed successfully during evaluation.")
            test_acc = 0.0
            avg_test_loss = 0.0
        else:
            test_acc = total_test_correct / total_test_samples
            avg_test_loss = total_test_loss_sum / total_test_samples
    
        # Process training statistics (stats_train is a list of dictionaries)
        num_train_samples_total = sum(s['num_samples'] for s in stats_train if s and 'num_samples' in s and s['num_samples'] > 0)
    
        if num_train_samples_total == 0:
            avg_train_loss = 0.0
            avg_train_ce_loss = 0.0
            avg_train_l_d_loss = 0.0
            avg_train_l_h_loss = 0.0
            print("Warning: No training samples were processed successfully for metrics.")
        else:
            # Weighted average of losses based on number of samples
            avg_train_loss = sum(s['loss'] * s['num_samples'] for s in stats_train if s and 'loss' in s) / num_train_samples_total
            avg_train_ce_loss = sum(s['ce_loss'] * s['num_samples'] for s in stats_train if s and 'ce_loss' in s) / num_train_samples_total
            avg_train_l_d_loss = sum(s['l_d'] * s['num_samples'] for s in stats_train if s and 'l_d' in s) / num_train_samples_total
            avg_train_l_h_loss = sum(s['l_h'] * s['num_samples'] for s in stats_train if s and 'l_h' in s) / num_train_samples_total
    
        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
    
        if loss is None:
            self.rs_train_loss.append(avg_train_loss) # Storing aggregated training loss
            # Optionally, store average test loss if needed for plotting/analysis
            if not hasattr(self, 'rs_test_loss'):
                self.rs_test_loss = []
            self.rs_test_loss.append(avg_test_loss)
        else:
            loss.append(avg_train_loss) # Or avg_test_loss depending on what `loss` parameter is for
    
        print(f"------------- Round {self.current_round} Evaluation Results -------------")
        print(f"Test Accuracy:          {test_acc:.4f}")
        print(f"Average Test Loss:      {avg_test_loss:.4f}")
        print("\nTraining Metrics (averaged over selected clients):")
        print(f"  Total Train Loss:     {avg_train_loss:.4f}")
        print(f"  CE Loss:              {avg_train_ce_loss:.4f}")
        print(f"  L_d Loss:             {avg_train_l_d_loss:.4f}")
        print(f"  L_h Loss:             {avg_train_l_h_loss:.4f}")
        print(f"  Num Training Samples: {num_train_samples_total}")
        print(f"  Num Test Samples:     {total_test_samples}")
        print("--------------------------------------------------------")
    
        # Store detailed train losses if needed
        if not hasattr(self, 'rs_train_ce_loss'):
            self.rs_train_ce_loss = []
            self.rs_train_l_d_loss = []
            self.rs_train_l_h_loss = []
    
        self.rs_train_ce_loss.append(avg_train_ce_loss)
        self.rs_train_l_d_loss.append(avg_train_l_d_loss)
        self.rs_train_l_h_loss.append(avg_train_l_h_loss)
    
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
    
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_test_loss = []
    
        clients_to_evaluate = self.selected_clients if self.selected_clients else self.clients
    
        for c in clients_to_evaluate:
            # Ensure current_round is passed to client for warmup factor calculation
            if hasattr(c, 'current_round') and hasattr(self, 'current_round'):
                c.current_round = self.current_round
    
            test_acc, test_loss, test_num, auc = c.test_metrics()  # 修改这里以接收所有4个返回值
    
            tot_correct.append(test_acc * 1.0)
            tot_test_loss.append(test_loss * 1.0)  # 使用实际的test_loss
            num_samples.append(test_num)
            tot_auc.append(auc * 1.0)
    
            if getattr(self, 'debug_mode', False) and test_num > 0:
                print(f"Client {c.id}: Test Acc: {test_acc/test_num:.4f}, Test Loss: {test_loss/test_num:.4f}, Test AUC: {auc:.4f}, Samples: {test_num}")
    
        return tot_correct, tot_test_loss, num_samples, tot_auc

    def save_results(self):
        """Override save_results from serverbase to include more information in filename"""
        file_path = None  # 在try外定义，确保在except中可以访问
        try:
            # 确保目录存在
            os.makedirs('results', exist_ok=True)
            
            # 确保所有属性都存在且有效
            model_str = getattr(self, 'model_str', 'unknown_local')
            global_model_type = getattr(self, 'global_model_type', 'unknown_global')
            
            # 构造文件名，使用 self.times 而不是 self.time
            file_name = f"{self.dataset}_{self.algorithm}_{model_str}_{global_model_type}_rounds{self.global_rounds}_{self.goal}_{self.times}"
            file_path = os.path.join('results', file_name + '.h5')
            
            print(f"Saving results to {file_path}")
            
            # 确保数据是numpy数组
            rs_test_acc = np.array(self.rs_test_acc) if self.rs_test_acc else np.array([])
            rs_train_loss = np.array(self.rs_train_loss) if self.rs_train_loss else np.array([])
            
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=rs_test_acc)
                hf.create_dataset('rs_train_loss', data=rs_train_loss)
            
            print(f"Successfully saved results to {file_path}")
            
        except Exception as e:
            error_msg = f"Error saving results"
            if file_path:
                error_msg += f" to {file_path}"
            error_msg += f": {str(e)}"
            print(error_msg)
            # 打印更详细的错误信息以帮助调试
            import traceback
            traceback.print_exc()
