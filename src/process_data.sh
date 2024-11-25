python generate_feature.py -d ICEWS14 -r rules_dict.json -l 1 2 3 -w 200 -p 20 -s train
python generate_feature.py -d ICEWS14 -r rules_dict.json -l 1 2 3 -w 200 -p 20 -s valid
python generate_feature.py -d ICEWS14 -r rules_dict.json -l 1 2 3 -w 200 -p 20 -s test
python generate_feature.py -d ICEWS18 -r rules_dict.json -l 1 2 3 -w 200 -p 48 -s train
python generate_feature.py -d ICEWS18 -r rules_dict.json -l 1 2 3 -w 200 -p 24 -s valid
python generate_feature.py -d ICEWS18 -r rules_dict.json -l 1 2 3 -w 200 -p 24 -s test
python generate_feature.py -d ICEWS05-15 -r rules_dict.json -l 1 2 3 -w 200 -p 48 -s train
python generate_feature.py -d ICEWS05-15 -r rules_dict.json -l 1 2 3 -w 200 -p 48 -s valid
python generate_feature.py -d ICEWS05-15 -r rules_dict.json -l 1 2 3 -w 200 -p 48 -s test
python generate_feature.py -d GDELT -r rules_dict.json -l 1 2 -w 200 -p 24 -s train
python generate_feature.py -d GDELT -r rules_dict.json -l 1 2 -w 200 -p 24 -s valid
python generate_feature.py -d GDELT -r rules_dict.json -l 1 2 -w 200 -p 24 -s test
# cross_dataset
python generate_feature.py -d ICEWS14 -r 0515_2_14_rules_dict.json -l 1 2 3 -w 200 -p 12 -s test
python generate_feature.py -d ICEWS18 -r 14_2_18_rules_dict.json -l 1 2 3 -w 200 -p 12 -s test
