## Requirements

- python==3.6
- torch==1.2
- scikit-learn==0.22.2

## How to run

### Ⅰ. Federated Learning on Source Cities

You can use the following three commands in three terminals, respectively

'''
!python main.py --rank 1 --data_dir "/data/city1.csv" --DARKL_checkpoint_path "/checkpoint/city_1/" --UTP_checkpoint_path "/checkpoint/city_1/"
!python main.py --rank 2 --data_dir "/data/city2.csv" --DARKL_checkpoint_path "/checkpoint/city_1/" --UTP_checkpoint_path "/checkpoint/city_1/"
!python main.py --rank 0 --data_dir "/data/city1.csv" --DARKL_checkpoint_path "/checkpoint/master/" --UTP_checkpoint_path "/checkpoint/master/"
'''

### Ⅱ. Knowledge Transferring on the Target City
'''
python transfer.py --data_dir "/data/target.csv" --DARKL_checkpoint_path "/checkpoint/master/" --UTP_checkpoint_path "/checkpoint/master/"
'''

