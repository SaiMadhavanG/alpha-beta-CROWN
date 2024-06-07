# Goal: To create a subclass of auto_LiRPA.BoundedModule, that inserts NAP Constraints whenever 'compute_bounds' is called

from auto_LiRPA import BoundedModule
import torch
import json

class NAPConstrainedBoundedModule(BoundedModule):
    """
        A subclass of auto_LiRPA.BoundedModule, that inserts NAP Constraints whenever 'compute_bounds()' is called
    """

    # Taking in path of the file containing NAPs in addition to normal arguments
    def __init__(self, model, global_input, nap_file_path, bound_opts=None,
                device='auto', verbose=False, custom_ops=None):
        
        super().__init__(model, global_input, bound_opts=bound_opts,
                device=device, verbose=verbose, custom_ops=custom_ops)

        self.nap_file_path = nap_file_path
        self.naps, self.naps_config = self.parse_naps(self.nap_file_path)
        self.label = None
        # self.layers_for_nap = self.get_layers_requiring_bounds()
        # TODO Hardcoded for now, change layer representation in NAP and translate
        self.layers_for_nap = ['/input', '/input.3', '/input.7', '/input.11']

        self.masks = self.create_masks()

        print("NAP Masks initialized")

    def set_label(self, label):
        # TODO exception handling
        self.label = str(label)

    def parse_naps(self, file_path):
        # TODO exception handling
        with open(file_path) as f:
            naps = json.load(f)
        config = naps['config']
        del naps['config']
        return naps, config
    
    def create_masks(self):
        """
            A method to create updation masks for interm/reference bounds.
            The output will be a dict having the following structure
            {
                label:
                    layer: (
                        Lower Bounds to be zeroed for activated neurons, # a boolean torch.tensor
                        Upper Bounds to be zeroed for deactivated neurons # a boolean torch.tensor
                        )
            }
        """
        labels = list(self.naps.keys())

        masks = {}


        for label in labels:
            masks[label] = {}
            # TODO fix this when NAP config changes
            for layer in range(self.naps_config['layers']):
                lb_mask = torch.zeros((self.naps_config['neurons_width'],), dtype=bool).to(self.device)
                ub_mask = torch.zeros((self.naps_config['neurons_width'],), dtype=bool).to(self.device)
                masks[label][self.layers_for_nap[layer]] = (lb_mask, ub_mask)

        for label in labels:
            for layer, neuron_idx in self.naps[label]["A"]["indices"]:
                masks[label][self.layers_for_nap[layer]][0][neuron_idx] = True
            for layer, neuron_idx in self.naps[label]["D"]["indices"]:
                masks[label][self.layers_for_nap[layer]][1][neuron_idx] = True

        return masks

    def compute_bounds(self, x=None, aux=None, C=None, method='backward', IBP=False,
            forward=False, bound_lower=True, bound_upper=True, reuse_ibp=False,
            reuse_alpha=False, return_A=False, needed_A_dict=None,
            final_node_name=None, average_A=False,
            interm_bounds=None, reference_bounds=None,
            intermediate_constr=None, alpha_idx=None,
            aux_reference_bounds=None, need_A_only=False,
            cutter=None, decision_thresh=None,
            update_mask=None, ibp_nodes=None, cache_bounds=False):
        
        if not self.label:
            raise Exception("Please set a label :(")
        
        # When there are no interm, reference or aux_reference bounds being passed, we can create a reference bound and set NAP constraints on that
        if not interm_bounds and not reference_bounds and not aux_reference_bounds:
            reference_bounds = {}
            for layer in self.layers_for_nap:
                lb = torch.full((self.naps_config['neurons_width'],), -float('inf')).to(self.device)
                lb[self.masks[self.label][layer][0]] = 0.
                ub = torch.full((self.naps_config['neurons_width'],), float('inf')).to(self.device)
                ub[self.masks[self.label][layer][1]] = 0.
                reference_bounds[layer] = (lb, ub)
        else:
            # If any of interm, reference or aux_reference bounds are passed, we apply the mask ans set the values to 0.
            if interm_bounds:
                for layer in self.layers_for_nap:
                    if layer in interm_bounds:
                        interm_bounds[layer][0].view((-1, self.naps_config['neurons_width']))[:, self.masks[self.label][layer][0]] = 0.
                        interm_bounds[layer][1].view((-1, self.naps_config['neurons_width']))[:, self.masks[self.label][layer][1]] = 0.
                    else:
                        raise Exception("I sincerely hope this error never occurs; interm")
            if reference_bounds:
                for layer in self.layers_for_nap:
                    if layer in reference_bounds:
                        reference_bounds[layer][0].view((-1, self.naps_config['neurons_width']))[:, self.masks[self.label][layer][0]] = 0.
                        reference_bounds[layer][1].view((-1, self.naps_config['neurons_width']))[:, self.masks[self.label][layer][1]] = 0.
                    else:
                        raise Exception("I sincerely hope this error never occurs; ref")
            if aux_reference_bounds:
                for layer in self.layers_for_nap:
                    if layer in aux_reference_bounds:
                        aux_reference_bounds[layer][0].view((-1, self.naps_config['neurons_width']))[:, self.masks[self.label][layer][0]] = 0.
                        aux_reference_bounds[layer][1].view((-1, self.naps_config['neurons_width']))[:, self.masks[self.label][layer][1]] = 0.
                    else:
                        raise Exception("I sincerely hope this error never occurs; aux_ref")
            
                    

        
        return super().compute_bounds(
            x , aux , C , method, IBP,
            forward, bound_lower , bound_upper , reuse_ibp,
            reuse_alpha, return_A, needed_A_dict,
            final_node_name, average_A,
            interm_bounds, reference_bounds ,
            intermediate_constr , alpha_idx ,
            aux_reference_bounds , need_A_only,
            cutter , decision_thresh ,
            update_mask , ibp_nodes , cache_bounds
        )