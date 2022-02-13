import numpy as np
import xgboost as xgb

if __name__ == "__main__":
    training_data = xgb.DMatrix('/Users/qiwangwq/dev/data/MQ2008/Fold1/train_dat.txt')
    testing_data = xgb.DMatrix('/Users/qiwangwq/dev/data/MQ2008/Fold1/test_dat.txt')
    # training_data = xgb.DMatrix('/Users/qiwangwq/dev/Fold1/train_dat.txt')
    # testing_data = xgb.DMatrix('/Users/qiwangwq/dev/Fold1/test_dat.txt')


    print(testing_data.num_col())
    print(testing_data.num_row())

    # the following are for version 1.5.2
    # param = {'max_depth':8, 'eta':0.1, 'objective':'rank:pairwise'}
    # model = xgb.train(param, training_data, num_boost_round=100, evals=[(testing_data, "testing_data")], verbose_eval=True)

    param = {'max_depth':6, 'eta':0.1, 'objective':'rank:pairwise', 'silent':1}
    model = xgb.train(param, training_data, num_boost_round=100, evals=[(testing_data, "testing_data")], verbose_eval=True)

    dump_list = model.get_dump()
    num_trees = len(dump_list)
    print(num_trees)

    model.save_model('/Users/qiwangwq/dev/gbdt-rs/examples/model.json')