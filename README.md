# Quality Control for OBSEA #

Scripts to apply quality control (QARTOD) to OBSEA's data. To insall:  

```bash

$ git clone https://github.com/obsea-upc/example_qc.git
$ cd example_qc
$ pip3 install -r requirements.txt
```

The code is shipped with two CTD datasets, to test the code:
 ```bash
 $ python3 qc.py -i OBSEA_SBE37_CTD_lite.csv -q qc_config/ctd_obsea.json -o test.csv -S qcout
 ```