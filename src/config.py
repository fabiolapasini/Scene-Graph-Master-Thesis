if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    
import os
import json


class Config(dict):
    def __init__(self, config_path):
        super().__init__()
        if isinstance(config_path, str):
            with open(config_path, 'r') as f:
                d = Config(json.load(f))
                for k,v in d.items():
                    self[k]=v
                self['PATH'] = os.path.dirname(config_path)
                # check input fields
                self.check_keys(self)
                        
            try: 
                import torch
                if torch.cuda.is_available() and len(self['GPU']) > 0:
                    self['DEVICE'] = torch.device("cuda")
                else:
                    self['DEVICE'] = torch.device("cpu")
            except:
                pass
        elif isinstance(config_path, dict):
            for k,v in config_path.items():
                self[k]=v
        else:
            raise RuntimeError('input must be str or dict')
            

    def check_keys(self, d:dict()):
        for key,value in d.items():
            if isinstance(value, dict):
                self.check_keys(value)
            else:    
                comment_key = '_'+key
                if comment_key in d.keys():
                    if d[key] not in d[comment_key]:
                        raise RuntimeError('value for',key,'should be one of',d[comment_key],'got',d[key])
            if isinstance(value, dict):
                self[key]= Config(value)


    def __dir__(self):
        return self.keys()
                

    def __getattr__(self, name):
        if self.get(name) is not None:
            return self[name]
        raise RuntimeError('key',name,'is not defined!')

        return None
    

    def __setattr__(self, name, value):
        self[name]=value
    

    def __get__(self, instance, owner):
        print('get')
        return self.value


    def __set__(self, instance, value):
        print('hello')
        self.value = float(value)


    def get_format_str_from_dict(self,name,iitem, indent=0):
        text = str()
        if isinstance(iitem, dict):
            text += 'dict:' + name + '\n'
            for key,value in iitem.items():
                text += self.get_format_str_from_dict(key,value,indent+2)
        else:
            for i in range(indent): text += ' '
            text += '{} {}\n'.format(name,iitem)
        return text
            
            
    def __repr__(self):
        text = str()
        for key, item in self.items():
            text +=self.get_format_str_from_dict(key,item)    
        return text


if __name__ == '__main__':    
    config = Config('../config_example.json')

    # print(config.dataset.dataset_type)
    print(config)

    '''
        _MODEL ['SGFN', 'SGPN']
        dict:MODEL
          N_LAYERS 2
          USE_SPATIAL False
          WITH_BN False
          USE_GCN True
          USE_RGB False
          USE_NORMAL False
          USE_CONTEXT True
          USE_GCN_EDGE True
          USE_REL_LOSS True
          OBJ_PRED_FROM_GCN True
          _GCN_TYPE ['TRIP', 'EAN']
          GCN_TYPE TRIP
          _ATTENTION ['fat']
          ATTENTION fat
          DROP_OUT_ATTEN 0.5
          multi_rel_outputs True
          feature_transform False
          point_feature_size 256
          edge_feature_size 256
          gcn_hidden_feature_size 512
          lambda_o 0.1
          NUM_HEADS 4
          _GCN_AGGR ['add', 'mean', 'max']
          GCN_AGGR max
          DIM_ATTEN 256
          _WEIGHT_EDGE ['BG', 'DYNAMIC', 'OCCU']
          WEIGHT_EDGE DYNAMIC
          w_bg 1.0
          NONE_RATIO 1.0
        VERBOSE True
        SEED 2020
        MAX_EPOCHES 40
        LR 0.001
        W_DECAY False
        AMSGRAD False
        _LR_SCHEDULE ['None', 'BatchSize']
        LR_SCHEDULE None
        GPU [0]
        SAVE_INTERVAL 100
        VALID_INTERVAL 5
        LOG_INTERVAL 10
        LOG_IMG_INTERVAL 100
        WORKERS 0
        _EDGE_BUILD_TYPE ['FC', 'KNN']
        EDGE_BUILD_TYPE KNN
        WEIGHTING True
        dict:dataset
          root ['.\\Data\\']
          selection
          ignore_scannet_rel True
          is_v2 True
          _label_file ['labels.instances.align.annotated.v2.ply', 'inseg.ply', 'cvvseg.ply']
          label_file labels.instances.align.annotated.v2.ply
          data_augmentation False
          num_points 128
          num_points_union 256
          disable_support_rel False
          with_bbox False
          discard_some False
          _dataset_type ['SGPN', 'SGFN']
          dataset_type SGPN
          load_cache False
          sample_in_runtime True
          sample_num_nn 2
          sample_num_seed 4
          class_choice []
          max_edges -1
          drop_edge 0.5
          drop_edge_eval 0.0
        PATH ..
        DEVICE cuda
    '''