import os
import json


class MpddSolver(object):
    CLSNAMES = ['brain']

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.label_dict={}
        self.read_csv('/ssd2/yangziteng/anomaly_detect/data/labels.csv')

    def read_csv(self, csv_file):
        with open(csv_file, 'r') as f:
            for line in f: 
                key,value=line.strip().split(',')
                self.label_dict[key]=value
            
    def run(self):
        #print(self.label_dict)
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['test']:
                cls_info = []
                img_names = os.listdir(f'{cls_dir}')
                for img_name in img_names:
                    img_name_id=img_name.split('.')[0]
                    is_abnormal = int(self.label_dict[str(int(img_name_id))])
                    info_img = dict(
                        img_path=f'{cls_dir}/{img_name}',
                        mask_path="",
                        cls_name=cls_name,
                        specie_name=img_name,
                        anomaly=is_abnormal,
                    )
                    cls_info.append(info_img)
                    if phase == 'test':
                        if is_abnormal:
                            anomaly_samples = anomaly_samples + 1
                        else:
                            normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    runner = MpddSolver(root='/ssd2/yangziteng/anomaly_detect/data/head_ct')
    runner.run()
