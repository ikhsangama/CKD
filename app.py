import os

from flask import Flask, render_template, request, redirect
import pymysql

import MyDir as my

connection = pymysql.connect(host="localhost", user="root", passwd="", database="knn")
cursor = connection.cursor()
# awalnya bagian demo bisa menerima semua atribut dan data yang tidak diisi diubah jadi NaN
# tapi kalau seperti itu, bagian
app = Flask(__name__)

# ######################################################
import numpy as np
import pandas as pd
import pymysql as pskl
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import time

connection = pskl.connect(host="localhost", user="root", passwd="", database="knn")
cursor = connection.cursor()


# Menampilkan semua array
# np.set_printoptions(threshold=np.nan)

#
# 2
#
def dataset(retrieve="all", id=1, target=1):  # hasil bertipe dataframe
    """
    parameters
    ----------
    retrieve: (all, numeric, polinom)
        all    : semua kolom
        numeric: hanya kolom numeric
        polinom: hanya kolom polinom
    id: (0, 1)
        0: tanpa kolom id
        1: dengan kolom id
    class: (0, 1)
        0: tanpa kolom class
        1: dengan kolom class

    """
    # connection = pskl.connect(host="localhost", user="root", passwd="", database="knn")
    # cursor = connection.cursor()
    if (retrieve == "all"):
        cols = ["id", "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
                "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]
        retrieve = "SELECT id, age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane, class FROM ckd_preprocessing3"
    elif (retrieve == "numeric"):
        cols = ["id", "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
                "class"]
        retrieve = "SELECT id, age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, class FROM ckd_preprocessing3"
    elif (retrieve == "polinom"):
        cols = ["id", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]
        retrieve = "SELECT id, rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane, class FROM ckd_preprocessing3"
    resolveall = cursor.execute(retrieve)
    rows_tupple = cursor.fetchall()
    data = pd.DataFrame(list(rows_tupple))
    data.columns = cols
    if (id == 0):
        data = data.drop(["id"], axis=1)
    if (target == 0):
        data = data.drop(["class"], axis=1)
    data = data.fillna(value=np.nan)  # mengubah missing value menjadi NaN
    return data


#
# 3
#
def outliers(df, k1=0.25, k3=0.75, multiply=1.5):
    """
    parameters:
    -----------
    df: input tipe dataframe, hanya menerima numeric

    mendeteksi data yang diluar batas bawah dan batas atas
    batas bawah = k1 - (k3-k1)*multiply
    batas atas = k3 + (k3-k1)*multiply
    """
    pencilan = df.apply(
        lambda x: (x < df[x.name].quantile(k1) - ((df[x.name].quantile(k3) - df[x.name].quantile(k1)) * multiply)) | (
                    x > df[x.name].quantile(k3) + ((df[x.name].quantile(k3) - df[x.name].quantile(k1)) * multiply)),
        axis=0)
    return pencilan


def outliers_removing(df, pencilan):
    """
    paramters:
    ----------
    df: input tipe dataframe
    pencilan: input tipe dataframe, bertipe boolean

    output:
    -------
    mengembalikan dataframe yang sudah menghilangkan semua outliers (semua baris yang berisi nilai pencilan true)
    """
    filtered_df = df[~(pencilan).any(axis=1)]
    return filtered_df


#
# 4.1
#
def encoding(df, cols=["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]):
    """
    paramters:
    ----------
    df: input tipe dataframe
    cols: input tipe array, menentukan kolom nominal mana yang akan di transformasi menjadi numerik

    output:
    ----------
    mengembalikan dataframe dengan kolom tertentu yang sudah di encode selain "NaN"

    """
    # python melakukan pass by reference, sehingga dibuat copy agar df sebelumnya tidak berubah

    copy_df = df.copy()
    for col in cols:
        a = copy_df[col].unique().tolist()
        c = [x for x in a if str(x) != 'nan']
        l = []
        i = 0
        for x in c:
            l.append(i)
            i += 1
        copy_df[col] = copy_df[col].replace(c, l)

    return copy_df


#
# 4.2
#
def missing_handling(df, cols=["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc",
                               "rbcc", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"],
                     method="mean"):
    """
    paramters:
    ----------
    df: input tipe dataframe
    cols: menentukan kolom numerik mana yang akan dilakukan penanganan missing value
    method: strategi penanganan missing value

    output:
    ----------
    mengembalikan dataframe dengan kolom yang sudah dilakukan penanganan "NaN"
    """

    copy_df = df.copy()
    imputer = Imputer(missing_values="NaN", strategy=method, axis=0)

    for col in cols:
        imputer = imputer.fit(copy_df[[col]])
        filledmissing_df = imputer.transform(copy_df[[col]])
        df_change = filledmissing_df.ravel()
        copy_df[col] = df_change

    return copy_df


#
# 4.3 NORMALISASI
#
from sklearn.preprocessing import MinMaxScaler


def normalizing(df, cols=["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
                          "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"], f_range=(0, 1)):
    """
    paramters:
    ----------
    df: input tipe dataframe
    cols: menentukan kolom numerik mana yang akan dilakukan penanganan missing value
    range: range normalisasi

    output:
    ----------
    mengembalikan dataframe dengan kolom yang sudah dinormalisasi
    """

    copy_df = df.copy()
    scale = MinMaxScaler(feature_range=f_range)
    for col in cols:
        normalization_array = scale.fit_transform(copy_df[[col]])
        df_change = normalization_array.ravel()
        copy_df[col] = df_change

    return copy_df

# Backward Elimination
import statsmodels.formula.api as sm


def backward_elimination(splitdf_x, splitdf_y, sl=0.05):
    sl_ = sl
    copy_df = splitdf_x.copy()
    reg_ols = sm.OLS(endog=splitdf_y, exog=copy_df).fit()
    pval = reg_ols.pvalues
    maks_pval = max(pval)

    if (maks_pval >= sl_):
        delete_col = pval[pval == maks_pval].index[0]
        drp = [delete_col]
        copy_df = copy_df.drop(drp, axis=1)
        copy_df = backward_elimination(copy_df, splitdf_y, sl_)

    else:
        copy_df = pd.concat([copy_df, splitdf_y], axis=1)
    return copy_df



#
# 6.1 Klasifikasi dengan kNN
#
import MyDir as my

#
# 7.1 Mencari nilai terbaik k pada kNN
#
from sklearn.model_selection import KFold

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def knn_kfold(df, kNN=5, k_fold=10, shuffle_=True, arr_akurasi=0):
    """
    Parameter:
    ----------
    df : Dataframe, data bertipe float / integer, tidak boleh ada NaN
    kNN : Nilai k-tetangga/data terdekat, default = 5
    k_fold : Jumlah fold / lipatan split training - test. Default = 5
    shuffle_ : Bentuk shuffle split jika diperlukan. Default = True

    Ouput:
    -------
    arr_akurasi : 1 untuk menghasilkan output array akurasi tiap fold, 0 untuk rata - rata akurasi. default = 0
    """

    kf = KFold(n_splits=k_fold, shuffle=shuffle_, random_state=49)
    akurasi = []
    precision = []
    recall = []
    sensitivity = []
    specificity = []
    confusion = []
    for train_index, test_index in kf.split(df):
        train_df = df.iloc[train_index]
        x_train = np.array(train_df.iloc[:, 1:-1])  # dimulai dari tanpa kolom id
        y_train = np.array(train_df["class"])
        test_df = df.iloc[test_index]
        x_test = np.array(test_df.iloc[:, 1:-1])  # dimulai dari tanpa kolom id
        y_test = np.array(test_df["class"])

        my_predictions = np.array([my.knn_predict(p, x_train, y_train, k=kNN) for p in x_test])
        #         confusion matrix
        tp, fn, fp, tn = confusion_matrix(y_test, my_predictions).ravel()
        confusion.append([tp, fn, fp, tn])
        akurasi.append((tp + tn) / (tp + tn + fp + fn) * 100)
        # precision.append(((tp) / (tp + fp)) * 100)
        # recall.append(((tp) / (tp + fn)) * 100)
        sensitivity.append((tp) / (tp + fn) * 100)
        specificity.append((tn) / (tn + fp) * 100)
    #         .confusion matrix

    if (arr_akurasi == 1):
        return confusion, akurasi, sensitivity, specificity
    else:
        return np.mean(akurasi), np.mean(sensitivity), np.mean(specificity)


def find_best_knn(df, start=3, finish=16, step=2, graph=True):
    akurasi = []
    maks = 0
    for i in range(start, finish, step):
        akurasi_k = knn_kfold(df, kNN=i, k_fold=10, shuffle_=True, arr_akurasi=0)
        akurasi.append([i, akurasi_k])
        if (maks < akurasi_k):
            maks = akurasi_k
            k = i
            print(k, maks)
        if (graph == True):
            x = [i[0] for i in akurasi]
            y = [i[1] for i in akurasi]
            plt.plot(x, y)
            plt.title("Mencari k Terbaik kNN")
            plt.xlabel("k")
            plt.ylabel("akurasi")
            plt.grid()
            plt.show
    return akurasi


# CLASS PUBLIC DATASTORE
class DataStore():
    terbaik = False
    dataset_df = dataset()
    target_df = dataset_df['class']
    dict_tahap_demo = {}

    cols = []
    cols_def = ["sg", "al", "bu", "sc", "sod", "hemo", "rbc", "htn", "dm", "appet"]
    cols_simpan = []

    simpan = False
    dict_tahap_simpan = {}
    dict_tahap_default = {
        '1': cols_def,
        '2': None,
        '3': "Transformasi Nominal Menjadi Numerik",
        '4': "Penanganan Missing Value",
        '5': "Normalisasi",
        '6': None,
        '7': 'Prediksi kNN, k= 5',
    }
    outlier_temp = False
    outlier_def = False
    outlier_simpan = False

    normalisasi_temp = False
    normalisasi_def = True
    normalisasi_simpan = False

    k_def = 5
    k_temp = 5
    k_simpan = None
    # SEMUA ATRIBUT


data = DataStore()


# .CLASS PUBLIC DATASTORE

@app.route('/')
def index():
    rule = request.url_rule.rule
    class_counts = data.dataset_df["class"].value_counts()
    return render_template('index.html', class_counts=class_counts, rule=rule,
                           title="Deteksi Penyakit Ginjal Kronis")


@app.route('/deteksi')
def deteksi():
    rule = request.url_rule.rule
    if(data.simpan):
        tahap = data.dict_tahap_simpan
        atribut = data.cols_simpan
    else:
        reset()
        tahap = data.dict_tahap_default
        atribut = data.cols_def

    return render_template('deteksi.html', title="Deteksi Pasien", rule=rule, tahap=tahap, atribut=atribut)


@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        nama = request.form['nama']
        l1 = []
        if (data.simpan == True):
            cols = data.cols_simpan
            kNN = data.k_simpan
        else:
            cols = data.cols_def
            kNN = data.k_def
        for i in cols:
            try:
                l1.append(float(request.form[i]))
            except:
                l1.append(request.form[i])

        # MENGAMBIL INPUT SEBAGAI DF DAN MENGGABUNGKAN DENGAN TRAINING DF ATRIBUT TERBAIK
        array_p = np.array(l1)
        p_df = pd.DataFrame(array_p.reshape(1, -1), columns=cols)
        dataset_df = dataset()
        cut_df_id = dataset_df["id"]
        cut_df_cols = dataset_df[cols]  # membuang kolom id dan kelas
        cut_df_class = dataset_df["class"]
        inputed_df = cut_df_cols.append(p_df, ignore_index=True)
        print("inputeddf")
        print(inputed_df)
        # .MENGAMBIL INPUT SEBAGAI DF DAN MENGGABUNGKAN DENGAN TRAINING DF ATRIBUT TERBAIK
        # TRANSFORMASI & NORMALISASI INPUT
        new_point, points = best_attribute_training(inputed_df, cols=cols)
        # .TRANSFORMASI & NORMALISASI INPUT
        outcomes = dataset_df["class"].values
        # points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
        # outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        print('newp')
        print(new_point)
        print('poin')
        print(points)
        print('outc')
        print(outcomes)
        hasil = my.knn_predict(new_point, points, outcomes, k=kNN)
        itr = len(cols)

        rule = request.url_rule.rule
    return render_template('output.html', hasil=hasil, nama=nama, itr=itr, atribut=cols, list=l1,
                           title="Hasil Deteksi", rule=rule)


def best_attribute_training(dataset, cols=['sg', 'al', 'bu', 'sc', 'sod', 'hemo', 'rbc', 'htn', 'dm', 'appet']):
    # return pertama yang akan diuji, return kedua data training
    df = dataset.copy()
    cut_df = df[cols]

    # OUTLIERS REMOVING
    if (data.simpan == True):
        if (data.dict_tahap_simpan.get('2') == True):
            df = cut_df.copy()
            daftar_numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
                                  'rbc']
            numeric_cols = [i for i in cols if i in daftar_numeric_cols]
            numeric_df = df[numeric_cols]
            pencilan_df = outliers(numeric_df)
            cut_df = outliers_removing(df, pencilan_df)
    # .OUTLIERS REMOVING

    # ENCODING
    daftar_cols_nominal = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    cols_nominal = [i for i in cols if i in daftar_cols_nominal]
    encoding_df = encoding(cut_df, cols=cols_nominal)
    # .ENCODING

    # MISSING HANDLING
    missing_handling_df = missing_handling(encoding_df, cols=cols)
    final_df = missing_handling_df
    # .MISSING HANDLING

    # NORMALIZING
    if (data.simpan == False):
        normalizing_df = normalizing(missing_handling_df, cols=cols)
        final_df = normalizing_df
    else:
        if (data.dict_tahap_simpan.get('4') == "Normalisasi"):
            normalizing_df = normalizing(missing_handling_df, cols=cols)
            final_df = normalizing_df
    # .NORMALIZING
    return final_df.values[-1].tolist(), final_df.iloc[:-1].values


@app.route('/pemodelan_langsung')
def pemodelan_langsung():
    data.dict_tahap_demo = {}
    rule = request.url_rule.rule
    data.dataset_df = dataset()
    data.cols = []
    return render_template("pemodelan_langsung.html", rule=rule, title="Pemodelan")


@app.route('/langsung1', methods=['GET','POST'])
def langsung1():
    if request.method == "POST":
        # TAHAP1
        cols = []
        if (request.form.get('age', False) != False):
            cols.append('age')
        if (request.form.get('bp', False) != False):
            cols.append('bp')
        if (request.form.get('sg', False) != False):
            cols.append('sg')
        if (request.form.get('al', False) != False):
            cols.append('al')
        if (request.form.get('su', False) != False):
            cols.append('su')
        if (request.form.get('bgr', False) != False):
            cols.append('bgr')
        if (request.form.get('bu', False) != False):
            cols.append('bu')
        if (request.form.get('sc', False) != False):
            cols.append('sc')
        if (request.form.get('sod', False) != False):
            cols.append('sod')
        if (request.form.get('pot', False) != False):
            cols.append('pot')
        if (request.form.get('hemo', False) != False):
            cols.append('hemo')
        if (request.form.get('pcv', False) != False):
            cols.append('pcv')
        if (request.form.get('wbcc', False) != False):
            cols.append('wbcc')
        if (request.form.get('rbcc', False) != False):
            cols.append('rbcc')

        # MULAI NOMINAL
        if (request.form.get('rbc', False) != False):
            cols.append('rbc')
        if (request.form.get('pc', False) != False):
            cols.append('pc')
        if (request.form.get('pcc', False) != False):
            cols.append('pcc')
        if (request.form.get('ba', False) != False):
            cols.append('ba')
        if (request.form.get('htn', False) != False):
            cols.append('htn')
        if (request.form.get('dm', False) != False):
            cols.append('dm')
        if (request.form.get('cad', False) != False):
            cols.append('cad')
        if (request.form.get('appet', False) != False):
            cols.append('appet')
        if (request.form.get('pe', False) != False):
            cols.append('pe')
        if (request.form.get('ane', False) != False):
            cols.append('ane')

        atribut_awal = 'Atribut awal: '+ ', '.join(cols)

        data.dict_tahap_demo = {}
        data.cols = []
        data.cols = cols
        data.dict_tahap_demo['1'] = atribut_awal

        #     TAHAP2 PEMBERSIHAN OUTLIER
        if request.form.get('pem_out', False) != False:
            data.outlier_temp = True
            data.dict_tahap_demo['2'] = "Pembersihan Outlier"

            daftar_num_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc",
                               "rbcc"]  # untuk cek outlier
            num_cols = [i for i in data.cols if i in daftar_num_cols]
            numeric_df = data.dataset_df[num_cols]
            pencilan_df = outliers(numeric_df)
            filtered_df = outliers_removing(data.dataset_df, pencilan_df)
            dataset_df = filtered_df
            data.dataset_df = dataset_df


        #     TAHAP3 TRANSFORMASI DATA
        data.dict_tahap_demo['3'] = "Transformasi Data"
        daftar_col_nominal = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]  # untuk transformasi
        col_nominal = [i for i in data.cols if i in daftar_col_nominal]
        col_nominal.append("class")
        cut_df_id = data.dataset_df["id"]
        cut_df_col = data.dataset_df[data.cols]
        cut_df_class = data.dataset_df["class"]
        cut_df = pd.concat([cut_df_id, cut_df_col, cut_df_class], axis=1)
        encoding_df = encoding(cut_df, cols=col_nominal)
        data.dataset_df = encoding_df


        #     TAHAP4 PENANGANAN MISSING VALUE
        data.dict_tahap_demo['4'] = "Penanganan Missing Value"
        missing_handling_df = missing_handling(data.dataset_df, cols=data.cols)
        data.dataset_df = missing_handling_df


        #     TAHAP 5 NORMALISASI
        if request.form.get('norm', False) != False:
            data.dict_tahap_demo['5'] = "Normalisasi"
            normalizing_df = normalizing(data.dataset_df, cols=data.cols)
            data.dataset_df = normalizing_df


        #   TAHAP 6 BACKWARD ELIMINATION
        tahapan = request.form['atribut']
        if tahapan!= "tanpa":
            if tahapan == "be_1":
                sl = 0.1
            elif tahapan == "be_05":
                sl = 0.05

            cut_df_id = data.dataset_df["id"]
            cut_df_cols = data.dataset_df[data.cols]
            cut_df_class = data.dataset_df["class"]
            backwarding_df = backward_elimination(cut_df_cols, cut_df_class,sl=sl)
            backward_df = pd.concat([cut_df_id, backwarding_df], axis=1)
            data.cols = list(backward_df)
            data.cols.remove('id')
            data.cols.remove('class')

            if tahapan == "be_1":
                atribut_akhir = "Atribut hasil Backward Elimination, α = 0.1: " + ', '.join(data.cols)
            elif tahapan == "be_05":
                atribut_akhir = "Atribut hasil Backward Elimination, α = 0.05: " + ', '.join(data.cols)

            data.dict_tahap_demo['6'] = atribut_akhir
            data.dataset_df = backward_df


        #   TAHAP 7 PREDIKSI
        data.k_temp = int(request.form['kNN'])
        data.dict_tahap_demo['7'] = "Prediksi kNN, k= " + str(data.k_temp)
        cut_df_id = data.dataset_df["id"]
        cut_df_col = data.dataset_df[data.cols]
        cut_df_class = data.dataset_df["class"]
        cut_df = pd.concat([cut_df_id, cut_df_col, cut_df_class], axis=1)
        np.set_printoptions(threshold=np.nan)
        print("cut")
        print(cut_df)

        start_time = time.time()
        confusion, akurasi, sensitifity, specificity = knn_kfold(cut_df, arr_akurasi=1, kNN=data.k_temp)
        waktu_validasi = time.time() - start_time

        m_akurasi = round(np.mean(akurasi), 3)
        m_sensitifity = round(np.mean(sensitifity), 3)
        m_specificity = round(np.mean(specificity), 3)

        return render_template("manual6.html", tahap=data.dict_tahap_demo, terbaik=data.terbaik,
                               df=data.dataset_df, shape=data.dataset_df.shape, waktu_validasi=waktu_validasi,
                               akurasi=akurasi,
                               sensitifity=sensitifity, specificity=specificity, m_akurasi=m_akurasi,
                               m_sensitifity=m_sensitifity,
                               m_specificity=m_specificity, rule='/pemodelan_langsung', title='Pemodelan')


@app.route('/pemodelan_pertahap')
def pemodelan_pertahap():
    data.dict_tahap_demo = {}
    rule = request.url_rule.rule
    data.cols = []
    return render_template("pemodelan_pertahap.html", rule=rule, title="Pemodelan Per-tahapan")


@app.route('/tahap1', methods=['GET', 'POST'])
def render_1():
    data.cols, data.dict_tahap_demo['1'] = demo_1send()
    data.dataset_df = dataset()
    if len(data.cols) == 0:
        return redirect("pemodelan_pertahap")
    df_id = data.dataset_df["id"]
    df_col = data.dataset_df[data.cols]
    df_class = data.dataset_df["class"]
    data.dataset_df = pd.concat([df_id, df_col, df_class], axis=1)

    return render_template("manual1.html", shape=data.dataset_df.shape, df=data.dataset_df, rule='/pemodelan_pertahap',
                           title='Pemodelan Per-tahapan')


def demo_1send():
    if request.method == "POST":
        cols = []
        if (request.form.get('age', False) != False):
            cols.append('age')
        if (request.form.get('bp', False) != False):
            cols.append('bp')
        if (request.form.get('sg', False) != False):
            cols.append('sg')
        if (request.form.get('al', False) != False):
            cols.append('al')
        if (request.form.get('su', False) != False):
            cols.append('su')
        if (request.form.get('bgr', False) != False):
            cols.append('bgr')
        if (request.form.get('bu', False) != False):
            cols.append('bu')
        if (request.form.get('sc', False) != False):
            cols.append('sc')
        if (request.form.get('sod', False) != False):
            cols.append('sod')
        if (request.form.get('pot', False) != False):
            cols.append('pot')
        if (request.form.get('hemo', False) != False):
            cols.append('hemo')
        if (request.form.get('pcv', False) != False):
            cols.append('pcv')
        if (request.form.get('wbcc', False) != False):
            cols.append('wbcc')
        if (request.form.get('rbcc', False) != False):
            cols.append('rbcc')

        # MULAI NOMINAL
        if (request.form.get('rbc', False) != False):
            cols.append('rbc')
        if (request.form.get('pc', False) != False):
            cols.append('pc')
        if (request.form.get('pcc', False) != False):
            cols.append('pcc')
        if (request.form.get('ba', False) != False):
            cols.append('ba')
        if (request.form.get('htn', False) != False):
            cols.append('htn')
        if (request.form.get('dm', False) != False):
            cols.append('dm')
        if (request.form.get('cad', False) != False):
            cols.append('cad')
        if (request.form.get('appet', False) != False):
            cols.append('appet')
        if (request.form.get('pe', False) != False):
            cols.append('pe')
        if (request.form.get('ane', False) != False):
            cols.append('ane')

        atribut_awal = 'Atribut awal: ' + ', '.join(cols)

        data.dict_tahap_demo = {}
        data.cols = []
        data.cols = cols

    return cols, atribut_awal


# PENANGANAN OUTLIER
@app.route('/tahap2')
def render_2():
    data.dataset_df, shape = demo_2send()
    return render_template("manual2.html", terbaik=data.terbaik, df=data.dataset_df,
                           shape=shape, rule='/pemodelan_manual', title='Pemodelan Per-tahapan')


def demo_2send():
    data.dict_tahap_demo['2'] = "Pembersihan Outlier"

    daftar_num_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc",
                       "rbcc"]  # untuk cek outlier
    num_cols = [i for i in data.cols if i in daftar_num_cols]
    numeric_df = data.dataset_df[num_cols]
    pencilan_df = outliers(numeric_df)
    filtered_df = outliers_removing(data.dataset_df, pencilan_df)
    dataset_df = filtered_df
    shape = dataset_df.shape
    return dataset_df, shape


# .PENANGANAN OUTLIER


# NOMINAL -> NUMERIK
@app.route('/tahap3')
def render_3():
    data.dataset_df = demo_3send()
    return render_template("manual3.html", terbaik=data.terbaik, df=data.dataset_df,
                           shape=data.dataset_df.shape, rule='/pemodelan_manual', title='Pemodelan Per-tahapan')


def demo_3send():
    data.dict_tahap_demo['3'] = "Transformasi Nominal Menjadi Numerik"
    daftar_col_nominal = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]  # untuk transformasi
    col_nominal = [i for i in data.cols if i in daftar_col_nominal]

    cut_df_id = data.dataset_df["id"]
    cut_df_col = data.dataset_df[data.cols]
    cut_df_class = data.dataset_df["class"]
    cut_df = pd.concat([cut_df_id, cut_df_col, cut_df_class], axis=1)

    encoding_df = encoding(cut_df, cols=col_nominal)
    return encoding_df


# .NOMINAL -> NUMERIK


# MISSING HANDLING
@app.route('/tahap4')
def render_4():
    data.dataset_df = demo_4send()
    return render_template("manual4.html", terbaik=data.terbaik, df=data.dataset_df, shape=data.dataset_df.shape,
                           rule='/pemodelan_manual', title='Pemodelan Manual')


def demo_4send():
    data.dict_tahap_demo['4'] = "Penanganan Missing Value"
    missing_handling_df = missing_handling(data.dataset_df, cols=data.cols)
    return missing_handling_df


# .MISSING HANDLING


# NORMALISASI
@app.route('/tahap5')
def render_5():
    data.dict_tahap_demo['5'] = "Normalisasi"
    normalizing_df = normalizing(data.dataset_df, cols=data.cols)
    data.dataset_df = normalizing_df
    return render_template("manual5.html", terbaik=data.terbaik, df=data.dataset_df, shape=data.dataset_df.shape,
                           rule='/pemodelan_manual', title='Pemodelan Manual')


# .NORMALISASI -> RATA2


# PREDIKSI kNN
@app.route('/manual6', methods=['GET', 'POST'])
def render_6():
    data.k_temp = demo_6send()

    data.dict_tahap_demo['6'] = "Prediksi kNN, k= " + str(data.k_temp)

    cut_df_id = data.dataset_df["id"]
    cut_df_col = data.dataset_df[data.cols]
    cut_df_class = data.dataset_df["class"]
    cut_df = pd.concat([cut_df_col, cut_df_class], axis=1)

    start_time = time.time()
    confusion, akurasi, sensitifity, specificity = knn_kfold(cut_df, arr_akurasi=1, kNN=data.k_temp)
    waktu_validasi = time.time() - start_time

    m_akurasi = round(np.mean(akurasi), 3)
    m_sensitifity = round(np.mean(sensitifity), 3)
    m_specificity = round(np.mean(specificity), 3)

    return render_template("manual6.html", tahap=data.dict_tahap_demo, terbaik=data.terbaik,
                           df=data.dataset_df, shape=data.dataset_df.shape, waktu_validasi=waktu_validasi,
                           akurasi=akurasi,
                           sensitifity=sensitifity, specificity=specificity, m_akurasi=m_akurasi,
                           m_sensitifity=m_sensitifity,
                           m_specificity=m_specificity, rule='/pemodelan_manual', title='Pemodelan Manual')


def demo_6send():
    if request.method == "POST":
        kNN = int(request.form['kNN'])

    return kNN


# .PREDIKSI kNN

@app.route('/simpan')
def simpan():
    data.simpan = True
    data.dict_tahap_simpan = data.dict_tahap_demo.copy()
    data.k_simpan = data.k_temp
    data.cols_simpan = data.cols
    return redirect("deteksi")


@app.route('/reset')
def reset():
    data.simpan = False
    data.cols = data.cols_def
    data.dict_tahap_default['1'] = "Atribut: " + ', '.join(data.cols_def)
    return redirect("deteksi")


if __name__ == '__main__':
    app.debug = True
    app.run()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
