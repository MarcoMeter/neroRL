import torch
class TruncateMemory:
    def __init__(self, model, model_config, memory_length, device):
        """ Truncates the memory of a recurrent or transformer model to the current memory length.

        Arguments:
            model {ActorCriticSharedWeights}: The model to be truncated. 
            mask {torch.tensor} -- Memory mask (None if the model is not transformer-based)
            model_config {dict}: Model config
            device (_type_): The device to be used.
        """
        self.obs = []
        if "transformer" in model_config:
            self.trxl_config = model_config["transformer"]
            self.memory_mask = torch.tril(torch.ones((self.trxl_config["memory_length"], self.trxl_config["memory_length"])), diagonal=-1)
        else:
            self.trxl_config = None
        self.memory_length = memory_length
        self.model_config = model_config
        self.model = model
        self.device = device
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def init_transformer_memory(self, *args, **kwargs):
        return self.model.init_transformer_memory(*args, **kwargs)
    
    def init_recurrent_cell_states(self, *args, **kwargs):
        return self.model.init_recurrent_cell_states(*args, **kwargs)
    
    def init_recurrent_cell(self, recurrence_config, model, device):
        hxs, cxs = model.init_recurrent_cell_states(1, device)
        if recurrence_config["layer_type"] == "gru":
            recurrent_cell = hxs
        elif recurrence_config["layer_type"] == "lstm":
            recurrent_cell = (hxs, cxs)
        return recurrent_cell
    
    def add_obs(self, vis_obs, vec_obs):
        """ Adds an observation to the memory.

        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation (None if not available)
            vec_obs {numpy.ndarray/torch.tensor} -- Vector observation (None if not available)
        """
        # Remove oldest observation if memory is full
        if len(self.obs) == self.memory_length:
            self.obs.pop(0)
        self.obs.append((vis_obs, vec_obs))
        
    def reset(self):
        """ Resets the memory. """
        self.obs = []
        
    def forward(self, vis_obs, vec_obs, _in_memory, _mask, indices):
        """ Truncates the memory to the current memory length.
        
        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation (None if not available)
            vec_obs {numpy.ndarray/torch.tensor} -- Vector observation (None if not available)
            _in_memory {torch.tensor} -- Not used
            _mask {torch.tensor} -- Not used
            indices {torch.tensor} -- Indices to select the positional encoding that matches the memory window (None of the model is not transformer-based)  
            
        Returns:
            {list} -- Policy: List featuring categorical distributions respectively for each policy branch
            {torch.tensor} -- Value function: Value
            {torch.tensor or tuple} -- Current memory representation or recurrent cell state (None if memory is not used)       
        """
        # Add new observation to memory
        self.add_obs(vis_obs, vec_obs)
        # Create new memory
        if "recurrence" in self.model_config:
            in_memory = self.init_recurrent_cell(self.model_config["recurrence"], self.model, self.device)
        if "transformer" in self.model_config:
            in_memory = self.model.init_transformer_memory(1, self.trxl_config["memory_length"], self.trxl_config["num_blocks"], self.trxl_config["embed_dim"], self.device)
        # Recompute the policy, value and memory with the truncated episode
        n = len(self.obs)
        for i in range(n):
            vis_obs, vec_obs = self.obs[i]
            if "recurrence" in self.model_config:
                policy, value, new_memory, _ = self.model(vis_obs, vec_obs, in_memory, None)
                in_memory = new_memory
            if "transformer" in self.model_config:
                mask = self.memory_mask[i].unsqueeze(0)
                policy, value, new_memory, _ = self.model(vis_obs, vec_obs, in_memory, mask, indices)
                in_memory[:, i] = new_memory
        return policy, value, new_memory, None