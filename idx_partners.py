# -*- coding: utf-8 -*-
"""idx_partners.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QnvTj7wsCHeL84wSUxXSVCFNg4sa4SHu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Baca file CSV
df = pd.read_csv('loan_data_2007_2014.csv', low_memory=False)

# Tampilkan 5 baris pertama dari dataframe
df.head()

df.shape

df.info() # melihat info data awal

"""drop kolom yg isinya semua missing value > drop missing value diatas 40% > imputasi missing value dibawah 40%"""

# drop semua kolom yg full dengan null
to_drop = df.isnull().sum().sort_values() # Menghitung jumlah nilai NaN di setiap kolom dan mengurutkannya
to_drop = to_drop[to_drop == df.shape[0]] # Memilih kolom-kolom yang seluruhnya berisi nilai NaN
to_drop = list(to_drop.index) # Mengubah indeks dari Series menjadi list yang berisi nama-nama kolom yang seluruhnya berisi nilai NaN

#check dimensi dataset
print('dimensi dataset sebelum drop = ',df.shape)

#drop feature yang semua isinya nilai null
df.drop(columns=to_drop,inplace=True)

#check dimensi dataset
print('dimensi dataset setelah drop = ',df.shape)

#check persentase missing values untuk setiap feature
# Menghitung jumlah nilai NaN di setiap kolom, mengurutkannya secara menurun, dan mereset indeks
mv = df.isnull().sum().sort_values(ascending=False).reset_index()
mv.columns = ['feature','null'] # Mengganti nama kolom
mv['%'] = round(mv['null']*100/(df.shape[0]),3) # Menghitung persentase nilai NaN di setiap kolom dan menyimpannya dalam kolom '%'
mv = mv[mv['%']>0] # Memilih hanya baris-baris yang memiliki persentase nilai NaN lebih dari 0%
mv

#filter feature yang punya null values > 40%
mv1 = list(mv['feature'][mv['%']>40])
print(mv1)
#filter feature yang null values < 40%
mv2 = list(mv['feature'][mv['%']<40])
print(mv2)

# drop feature yang punya null values >40%
df.drop(columns=mv1,inplace=True)

#membagi data numerik dan kategori
numerik = []
categories = []
for i in mv2:
  if (df[i].dtype == 'object') or (df[i].dtype =='category'):
    categories.append(i)
  else:
    numerik.append(i)

"""### cleaning data numerik"""

#imputation
# melakukan imputasi ke feature numerik dengan median karena distribusi data lebuh banyak skewed dan mempunyai outlier
for i in numerik:
  df[i].fillna(df[i].median(),inplace=True)

"""### cleaning data kategorik"""

kategori_cols = df.select_dtypes(include=['object', 'category']).columns

#membuat tabel kategori dan melihat yang mana saya memiliki nilai uniq yg banyak dan akan dihapus
df[kategori_cols].describe().transpose()

#melihat isi kolom
for i in kategori_cols:
  print('-----'*10)
  print(i)
  print(df[i].value_counts())

#mengganti none dan any di kolom home ownership
df['home_ownership'] = df['home_ownership'].replace({'NONE': 'OTHER', 'ANY': 'OTHER'})

# Membuat kolom 'target' pada DataFrame 'df' dengan nilai 'good' atau 'bad'
df['status'] = np.where(
    (df['loan_status'] == 'Charged Off') |  # Jika status pinjaman adalah 'Charged Off'
    (df['loan_status'] == 'Default') |      # atau status pinjaman adalah 'Default'
    (df['loan_status'] == 'Late (31-120 days)') |  # atau status pinjaman adalah 'Late (31-120 days)'
    (df['loan_status'] == 'Late (16-30 days)') |   # atau status pinjaman adalah 'Late (16-30 days)'
    (df['loan_status'] == 'Does not meet the credit policy. Status:Charged Off'),  # atau status pinjaman adalah 'Does not meet the credit policy. Status:Charged Off'
    'bad',  # Maka nilai 'target' adalah 'bad'
    'good'  # Selain itu, nilai 'target' adalah 'good'
)

#drop kolom yang memiliki nilai uniq banyak,yg cuma memiliki 1,grade dan sub grade sama jadi pilih grade,loan status sudah diganti dengan feature target
df.drop(columns=['Unnamed: 0','emp_title','url','title','zip_code','addr_state','sub_grade','loan_status','application_type'],inplace=True)

cleaned_cat = df.select_dtypes(include=['object', 'category']).columns

#melihat kolom categories yang sudah dibersihkan
df[cleaned_cat].describe().transpose()

#mengisi missing value dengan nilai modus
df['emp_length'] = df['emp_length'].fillna('10+ years')
df['last_pymnt_d'] = df['last_pymnt_d'].fillna('Jan-16')
df['last_credit_pull_d'] = df['last_credit_pull_d'].fillna('Jan-16')
df['earliest_cr_line'] = df['earliest_cr_line'].fillna('Oct-00')

"""mengganti data tipe tanggal"""

from datetime import datetime as dt
df['issue_d'] = pd.to_datetime(df['issue_d'].apply(lambda x: dt.strptime(x, '%b-%y')))
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'].apply(lambda x: dt.strptime(x, '%b-%y')))
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'].apply(lambda x: dt.strptime(x, '%b-%y')))
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'].apply(lambda x: dt.strptime(x, '%b-%y')))

# mengganti tanggal tahun yang salah ketika perubahan ke tipe datetime pada kolom earliest cr line
def correct_year(date):
    if date.year > 2011:  # misalnya, jika tahun lebih besar dari 2011 dianggap tidak sesuai
        # tahun yang lebih dari 2011 akan dikurangi 100 karena pada analisis sebelumnya tahunnya ada yang 2044,2045,dst
        return date.replace(year=date.year - 100)
    return date

# Terapkan fungsi untuk memperbaiki tahun yang tidak sesuai
df['earliest_cr_line'] = df['earliest_cr_line'].apply(correct_year)

df.isnull().sum()

df.info()

"""# Exploratory Data Analysis

## univariate analysis

### categorical
"""

categories = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype == 'category']

plt.figure(figsize=(12, 11))
for i in range(min(len(categories), 9)):
    plt.subplot(3, 3, i + 1)
    # Urutkan nilai dari tertinggi ke terendah
    sorted_values = df[categories[i]].value_counts().index # Select the i-th category
    # Buat palet warna
    palette = sns.color_palette("viridis", len(sorted_values))
    # Buat countplot tanpa menggunakan hue
    sns.countplot(data=df, x=categories[i], order=sorted_values) # Plot the i-th category
    # Tetapkan warna secara manual
    for bar, color in zip(plt.gca().patches, palette): # Get the current axes
        bar.set_color(color)
    plt.title(f'Distribution of {categories[i]}', fontsize=14) # Set title for the i-th category
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

"""- pada kolom term jangka waktu cicilannya didominasi oleh 36 month
- pada kolom grade didominasi oleh b dan c
- pada kolom emp length didominasi oleh peminjam yang sudah lama bekerja lebih dari 10 tahun
- pada kolom home ownership didominasi oleh mortage
- pada kolom verification status didominasi oleh verified
- pada kolom payment plant didominasi oleh label n
- pada kolom distribution of purpose didominasi oleh debt consolidation
- pada kolom initial list status ddominasi oleh lbel f
- pada kolom status yaitu yang akan dijadikan target didominasi oleh good

### numerical
"""

numerics = [col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype == 'float64']

plt.figure(figsize=(15,15))
for i in range(0, len(numerics)):
    plt.subplot(8, 4, i+1)
    sns.boxplot(x=df[numerics[i]], color='royalblue')
    plt.xlabel(numerics[i])
    plt.tight_layout()

"""- dilihat dari data numerik kebanyakan distribusinya adalah skewed kanan dengan outlier

### DateTime
"""

dmy = ['issue_d','last_pymnt_d','last_credit_pull_d','earliest_cr_line']
for col in dmy:
    df[col] = pd.to_datetime(df[col])
    df[col+'_year'] = df[col].dt.year

plt.figure(figsize=(15, 8))
for i, col in enumerate(dmy):
    if col + '_year' in df.columns:
        plt.subplot(2, 2, i + 1)
        sns.countplot(x=df[col + '_year'], hue=df[col + '_year'], palette='viridis', legend=False)
        plt.xlabel(col + ' Year')
        plt.ylabel('Count')
        plt.title('Distribution of ' + col + ' Year')
        plt.xticks(rotation=90)
        plt.tight_layout()

plt.show()

"""- data pada tipe datetime menunjukkan bahwa setiap tahun mengalami peningkatan jumlah

## Bivariate analysis

### categorical
"""

plt.figure(figsize=(15, 8))
for i in range(min(len(categories), 8)):  # Batas maksimal subplot yang akan ditampilkan adalah 8
    plt.subplot(2, 4, i + 1)
    sorted_values = df[categories[i]].value_counts().index
    sns.countplot(data=df, x=categories[i], hue='status', order=sorted_values, palette="viridis")
    plt.xlabel(categories[i])
    plt.xticks(rotation=90)
    plt.tight_layout()

plt.show()

"""- dilihat dari data diatas bahwa peminjam yang bad kebanyakan meminjam dengan jangka waktu 36 bulan, berada pada grade c, rumah mereka masih menyewa dan juga tujuan mereka meminjam adalah untuk debt consolidation
- kalau dilihat dari yang good debitur kebanyakan mereka meminjam dengan jangka waktu 36 bulan kemudian status mereka ada pada  grade b, mempunyai rumah yg statusnya mortage dan mereka juga bekerja lebih dari 10 tahun kemudian tujuan mereka hutang adalah untuk debt consolidation yang didanai secara fractional

### numerical
"""

df_sample=df.sample(1000,random_state=1)
plt.figure(figsize=(15,20))
for i in range(0, len(numerics)):
    plt.subplot(8, 4, i+1)
    sns.boxplot(y=df_sample[numerics[i]], x=df['status'])
    plt.xlabel(numerics[i])
    plt.tight_layout()

"""- berdasarkan data diatas dapat dilihat bahwa peminjam yang gagal bayar atau bad memiliki int rate(suku bunga) yang tinggi,total rec late fee yang tinggi, recoveries yang tinggi dan dari semua data diatas distribusi data terhadap fitur target adalah skewness kekanan

### datetime
"""

plt.figure(figsize=(15, 8))
for i, col in enumerate(dmy):
    if col + '_year' in df.columns:
        plt.subplot(2, 2, i + 1)
        sns.countplot(x=df[col + '_year'], hue='status', data=df, palette='viridis')
        plt.xlabel(col + ' Year')
        plt.ylabel('Count')
        plt.title(f'Distribution of {col} Year by Status')
        plt.xticks(rotation=90)
        plt.tight_layout()

plt.show()

"""- terjadi peningkatan dari segi peminjam yang gagal bayar dan juga yang berhasil bayar pada setiap tahunnya dan puncaknya yaitu pada tahun 2000 kalau dilihat pada kolom earliest cr line

## Multivariate analysis
"""

numeric_df = df.select_dtypes(include=[float, int])

# Hitung matriks korelasi
correlation_matrix = numeric_df.corr()

# Buat peta panas
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='viridis')
plt.show()

# menghapus kolom yang tidak digunakan pada pembuatan model dan juga menghapus kolom yang tidak
# mempunyai korelasi dan menghapus kolom year karena sudah ada kolom date
df.drop(columns=['id','member_id','collections_12_mths_ex_med','policy_code','acc_now_delinq','tot_coll_amt','pymnt_plan','issue_d_year'
,'last_pymnt_d_year','last_credit_pull_d_year','earliest_cr_line_year'],inplace=True)

"""## insight"""

df_counts = df['status'].value_counts()
df_percentages = df['status'].value_counts(normalize=True) * 100

an = pd.DataFrame({'status': df_counts.index,'total': df_counts.values,'persen': df_percentages.values
})

# Atur ulang indeks untuk urutan yang lebih baik
an = an.reset_index(drop=True)

an

palette = sns.color_palette("viridis", len(an))

# Buat pie chart dengan warna viridis
plt.pie(an['total'], labels=an['status'], autopct='%.3f%%', colors=palette)
plt.show()

df.groupby('status').agg({'funded_amnt':'sum','total_pymnt':'sum'}).reset_index()

"""Perusahaan loss sebesar $760,916,150 karena 11.192% peminjam tidak mampu bayar"""

re = df.groupby(['status','grade','purpose']).agg({'recoveries':'mean'}).reset_index().sort_values(['recoveries'],ascending=False)
re.columns = ['status','grade','purpose','avg_recoveries']
re[re['status']=='bad']

plt.figure(figsize=(15,5))
sns.barplot(x='grade',y='avg_recoveries',hue='purpose',data=re[re['status']=='bad'],palette='viridis')
plt.title('Rata-rata recoveries peminjam yang gagal bayar berdasarkan grade dan purpose',fontsize=14)
plt.legend(bbox_to_anchor=(1,1))
plt.show()

"""Jika peminjam yang gagal bayar ditinjau dari feature grade, purpose, dan rata-rata nilai recoveries, maka peminjam yang berada pada kategori grade G dengan tujuan untuk energi terbarukan perlu dipertimbangkan untuk ditolak pengajuan kreditnya karena memiliki risiko kredit gagal bayar. begitu pula dengan peminjam yang berada pada kategori grade F dengan tujuan peminjaman wedding juga perlu dipertimbangkan untuk ditolak pengajuan kreditnya karena mimiliki risiko gagal bayar. Begitu pula pada grade B dengan tujuan peminjaman house memiliki resiko kredit gagal bayar sehingga dapat dipertimbangkan untuk menolak pengajuan kreditnnya."""

df.shape

"""# data preparation tanpa balacing data target"""

from pycaret.classification import setup

clf1 = setup(data=df,
             target='status',
             fix_imbalance=False,
             train_size=0.8,
             numeric_imputation='median')

"""# Modeling 1"""

from pycaret.classification import compare_models

# Melatih dan membandingkan beberapa model klasifikasi
best_model = compare_models(['lr','dt','rf','lightgbm'])

"""# Evaluation 1"""

from pycaret.classification import plot_model
# melakukan evaluasi terhadap model yang terbaik
# Plot confusion matrix
plot_model(best_model, plot='confusion_matrix')

plot_model(best_model, plot='auc')

"""# Data Preprocessing dengan balacing data target"""

from pycaret.classification import setup

clf2 = setup(data=df,
             target='status',
             train_size=0.8,
             fix_imbalance=True,
             numeric_imputation='median',
             normalize=True,
             normalize_method='minmax',
             encoding_method='onehot',
             )

"""# Modeling 2"""

from pycaret.classification import compare_models

# Melatih dan membandingkan beberapa model klasifikasi
best_model2 = compare_models(['lr','dt','rf','lightgbm'])

"""# Evaluation 2"""

from pycaret.classification import plot_model
# melakukan evaluasi terhadap model yang terbaik
# Plot confusion matrix
plot_model(best_model2, plot='confusion_matrix')

plot_model(best_model2, plot='auc')

plot_model(best_model2, plot='feature')

"""# tes prediksi"""

from pycaret.classification import predict_model  # Import the predict_model function
predict_model(best_model2)