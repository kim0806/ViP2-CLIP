import os
import json


class MVTecSolver(object):
    CLSNAMES = [
        'pcb'
    ]

    def __init__(self, root='/ssd2/yangziteng/anomalyclip/clip/data'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            #实力名
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                #所有图片名
                species = os.listdir(f'{cls_dir}/{phase}')
                #判断是否为缺陷类
                for img_name in species:
                    is_abnormal = True 
                    #图片及ground路径+实例名+训练还是测试+异常标签
                    info_img = dict(
                        img_path=f'{cls_name}/{phase}/{img_name}',
                        mask_path=None,
                        cls_name=cls_name,
                        specie_name='defect',
                        anomaly=1,
                    )
                    cls_info.append(info_img)
                #训练or测试/实例/图片
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = MVTecSolver()
    runner.run()
