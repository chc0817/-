import numpy as np
import pandas as pd


def information_describe(fi):

    df = pd.read_csv(fi, header=0, index_col=None)
    print(df.head(5))
    cID = df["ID"].tolist()
    uq_cID = list(set(cID))
    uq_cID.sort()
    print(len(uq_cID))

    inform_stats = np.zeros((len(uq_cID), 6))
    for i, id in enumerate(uq_cID):
        # extract information for current id
        cid_df = df.loc[df["ID"] == id, :]

        # number of invoice 
        this_invoice = cid_df["invoice"]
        this_invoice = set(this_invoice.tolist())

        invoice_num = len(this_invoice)
        inform_stats[i, 0] = invoice_num


        # total of price
        this_price = cid_df["price"].values
        total_price = this_price.sum()
        inform_stats[i, 1] = total_price


        # number of negative price
        nprice_num = sum(this_price < 0)
        inform_stats[i, 2] = nprice_num

        # number of invoice
        inform_stats[i, 3] = invoice_num

        # number of invalid invoice
        this_status = cid_df["status"].values
        invalid_invoice_num = sum(this_status == 0)
        inform_stats[i, 4] = invalid_invoice_num
        
        # tax and price in total
        this_tax_price = cid_df["tax_price"].values
        tax_price_sum = this_tax_price.sum()
        inform_stats[i, 5] = tax_price_sum

    return uq_cID, inform_stats

def merge_inform(in_uq_cID, out_uq_cID, in_inform_stats, out_inform_stats):

    num_record = len(in_uq_cID)
    features = np.zeros((num_record, 14))

    for i, id in enumerate(in_uq_cID):

        index_in_out_inform = out_uq_cID.index(id)
        
        # number of income inovice
        features[i, 0] = in_inform_stats[i, 0]

        # number of outcome inovice
        features[i, 1]  = out_inform_stats[index_in_out_inform, 0] 

        # total price of income voice
        features[i, 2] = in_inform_stats[i, 1]

        # total price of outcome voice
        features[i, 3] = out_inform_stats[index_in_out_inform, 1]

        # total number of invoice with negative price
        features[i, 4] = in_inform_stats[i, 2] + out_inform_stats[index_in_out_inform, 2]

        # total number of invoice
        features[i, 5] = in_inform_stats[i, 3] + out_inform_stats[index_in_out_inform, 3]

        # total number of invalid invoice
        features[i, 6] = in_inform_stats[i, 4] + out_inform_stats[index_in_out_inform, 4]

        # tax and price in total of income invoice
        features[i, 7] = in_inform_stats[i, 5]
        
        # tax and price in total of outcome invoice
        features[i, 8] = out_inform_stats[index_in_out_inform, 5]

        # RM
        features[i, 9] = features[i, 5]

        # LT
        features[i, 10] = features[i, 2] + features[i, 3]

        # WD 
        features[i, 11] = (features[i, 5] - (features[i, 4] + features[i, 6])) / (features[i, 5] + 1e-10)

        # YL
        features[i, 12] = (features[i, 8] - features[i, 7]) / (features[i, 8] + features[i, 7] + 1e-10)

        #企业估计毛利率ML
        features[i ,13] = (features[i, 3] - features[i, 2]) / features[i, 3]
        
    return in_uq_cID, features

incomes_fi = "tastin.csv"

in_uq_cID, in_inform_stats = information_describe(incomes_fi)

outcomes_fi = "tastout.csv"
out_uq_cID, out_inform_stats = information_describe(outcomes_fi)


cID_, features = merge_inform(in_uq_cID, out_uq_cID, in_inform_stats, out_inform_stats)

features[:,9]=features[:,9]/np.sum(features[:,9])
features[:,10]=features[:,10]/(2*np.sum(features[:,10]))
#市场活跃指数HY
#features[:, 14] = features[:, 10]/features[:, 11]

features_df = pd.DataFrame(data=features, index=cID_)

features_df.to_csv("tast_data_features.csv", header=False, index=True)
print()




