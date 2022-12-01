import pandas as pd
import pickle


def read_data(data_path):
    dataset = pd.read_csv(data_path)
    print(dataset.shape)
    return dataset


def complete_missing_col(test_df):
    train_data1 = pd.read_csv('./data/train.csv')
    train_data2 = fill_NaNs(train_data1)
    train_data = create_dummies(train_data2)
    train_data.drop('SalePrice', axis=1, inplace=True)
    missing_cols1 = set(train_data.columns) - set(test_df.columns)
    # missing_cols = missing_cols.tolist()
    # Add a missing column in test_df set with default value equal to 0
    for c in missing_cols1:
        test_df[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test_with_train_columns = test_df[train_data.columns]
    return test_with_train_columns





def fill_NaNs(df):
    df.drop('Id', axis=1, inplace=True)  # removing Id feature (will not give any info on the price just fifo).
    df['PoolQC'] = df['PoolQC'].fillna('NA')  # No pool
    df['MiscFeature'] = df['MiscFeature'].fillna('NA')  # no special element in the house.
    df['Alley'] = df['Alley'].fillna('NA')  # not access to alley
    df['Fence'] = df['Fence'].fillna('NA')  # no fence
    # same thing we will do to FireplaceQu, LotFrontage
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')  # no fireplace in the house.
    df['LotFrontage'] = df['LotFrontage'].fillna(0.)  # there is no front area.
    # and for the Garage missing houses and the Basement missing houses.
    df.fillna({'GarageType': 'NA', 'GarageFinish': 'NA', 'GarageQual': 'NA', 'GarageCond': 'NA', 'GarageYrBlt': .0},
              inplace=True)
    df.fillna({'BsmtExposure': 'NA', 'BsmtQual': 'NA', 'BsmtFinType2': 'NA', 'BsmtCond': 'NA', 'BsmtFinType1': 'NA'},
              inplace=True)
    df['MasVnrType'] = df['MasVnrType'].fillna('NA')  # No Masonry veneer type
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0.)  # No Masonry area.
    df['Electrical'] = df['Electrical'].fillna(
        df['Electrical'].mode()[0])  # we will replace the NaNs with the median=SBrkr
    return df


def create_dummies(df):
    df = pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True,
                        dtype=None)
    return df


def process_data(data_path):
    raw_data = read_data(data_path)
    dataset = fill_NaNs(raw_data)
    final_data = create_dummies(dataset)
    print(final_data.shape)
    final_data.columns.tolist()
    return final_data.dropna()


if __name__ == '__main__':
    with open('rf.pkl', 'rb') as f:
        model = pickle.load(f)
    precessed_data = process_data('./data/test.csv')
    precessed_data = complete_missing_col(precessed_data)
    results = model.predict(precessed_data)
    print()
