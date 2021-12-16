#Import pandas library
import pandas as pd

#Import dataset and chech main features

df = pd.read_csv(".../persona.csv")

df.head()
df.info()
df.shape
df.tail()
df.isnull().sum()
df.columns
df.describe().T

#How many unique SOURCE are there? What are their frequencies?

df["SOURCE"].nunique()
df["SOURCE"].unique()
df["SOURCE"].value_counts()

#How many unique PRICEs are there?

df["PRICE"].nunique()
df["PRICE"].unique()

#How many sales were made from which PRICE?

df["PRICE"].value_counts()

#How many sales from which country?

df["COUNTRY"].value_counts()

#How much was earned in total from sales by country?

df.groupby("COUNTRY")["PRICE"].sum()

#What are the sales numbers by SOURCE types?

df.groupby("SOURCE")["PRICE"].count()

#What are the PRICE averages by country?

df.groupby("COUNTRY")["PRICE"].mean()

#What are the PRICE averages by SOURCEs?

df.groupby("SOURCE")["PRICE"].mean()

#What are the PRICE averages in the COUNTRY-SOURCE breakdown?

df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean()

#Average earnings by COUNTRY, SOURCE, SEX, AGE


df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"})

#Sort by Price in descending order

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)



#Set indexes as column

agg_df = agg_df.reset_index()

#Change type of  "AGE" variable to categorical, bin values and add as new column

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 66],
                           labels=["0_18", "19_23", "24_30", "31_40", "41_66"])

#Identify new level-based customers (personas). and as column


agg_df["customer_level_based"] = [
    value[0].upper() + "_" + value[1].upper() + "_" + value[2].upper() + "_" + value[5].upper() for value in
    agg_df.values]

agg_df = agg_df[["customer_level_based", "PRICE"]]

agg_df = agg_df.groupby("customer_level_based")["PRICE"].mean().reset_index()

#Segment new customers (personas).

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})
agg_df[agg_df["SEGMENT"] == "C"].agg({"PRICE": ["mean", "max", "sum"]})

#Classify new customers by segment and predict how much money they bring.

def predict_income(new_user):
    return agg_df[agg_df["customer_level_based"]==new_user]

predict_income("TUR_ANDROID_FEMALE_31_40")
predict_income("FRA_IOS_FEMALE_31_40")
