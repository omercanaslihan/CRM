##RFM Analizi ile Müşteri Segmentasyonu##

#İş Problemi#
#Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama
#stratejileri belirlemek istiyor.Buna yönelik olarak müşterilerin davranışları tanımlanacak ve
#bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

#Veri Seti Hikayesi
#Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
#olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

#Görev 1: Veriyi Anlama ve Hazırlama
import pandas as pd
import datetime as dt
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.width", 1000)

#Adım1: flo_data_20K.csvverisiniokuyunuz.Dataframe’inkopyasınıoluşturunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

#Adım2: Verisetinde
#a. İlk 10 gözlem,
#b. Değişken isimleri,
#c. Betimsel istatistik,
#d. Boş değer,
#e. Değişken tipleri, incelemesi yapınız.

df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.info()

#Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
#Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

#Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

#Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"master_id": "count",
                                 "total_order": "sum",
                                 "total_customer_value": "sum"})

#Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.sort_values("total_customer_value", ascending=False).head(10)

#Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values("total_order", ascending=False).head(10)

#Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_prep(dataframe):
    #Toplam sipraiş ve harcama tutarı
    dataframe["total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_customer_value"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    #Tarihleri datetimea çevirme
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

data_prep(df)


#Görev 2: RFM Metriklerinin Hesaplanması

#Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

#Recency: Müşterinin son alışveriş zamanı üzerinden geçen zaman
#Frequency: Müşterinin alışveriş yapma sıklığı
#Monetary: Müşterinin kazandırdığı para

#Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
last_date = df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

df.groupby("master_id").agg({"last_order_date": lambda x: (today_date-x.max()).days, "total_order": lambda x: x.max(), "total_customer_value": lambda x: x.max()})

#Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date-x.max()).days, "total_order": lambda x: x.max(), "total_customer_value": lambda x: x.max()})
rfm.reset_index(inplace=True)
rfm.head()

#Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm.columns = ["master_id", "recency", "frequency", "monetary"]
rfm.head()

#Görev 3: RF Skorunun Hesaplanması
#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

#Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

#Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = rfm["recency_score"].astype("str") + rfm["frequency_score"].astype("str")

#Görev 4: RF Skorunun Segment Olarak Tanımlanması
seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions",
}

rfm["SEGMENT"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

#Görev 5: Aksiyon Zamanı !
#Adım 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm.groupby("SEGMENT").agg({'recency': ["mean", "count"],
                            'frequency': ["mean", "count"],
                            'monetary': ["mean", "count"]})

#Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.
#a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde.
# Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçmek isteniliyor.
# Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
loyal_champ_woman = pd.merge(rfm, df)
loyal_champ_woman = loyal_champ_woman[((loyal_champ_woman["SEGMENT"] == "champions") | (loyal_champ_woman["SEGMENT"] == "loyal_customers"))
                  & loyal_champ_woman["interested_in_categories_12"].str.contains("KADIN")][["master_id", "SEGMENT", "interested_in_categories_12"]]
loyal_champ_woman["master_id"].to_csv("l_c_w.csv", index=False)

#Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan
#ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor.
#Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.
seg = pd.merge(rfm, df)
seg = seg[((seg["SEGMENT"] == "hibernating") | (seg["SEGMENT"] == "at_risk") | (seg["SEGMENT"] == "cant_loose") | (seg["SEGMENT"] == "new_customers")) &
          ((seg["interested_in_categories_12"].str.contains("ERKEK")) | (seg["interested_in_categories_12"].str.contains("COCUK")))]
seg["master_id"].to_csv("e_c_c.csv", index=False)
