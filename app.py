import streamlit as st
import pandas as pd
from pycaret.classification import predict_model, load_model



# Muat model yang telah disimpan
model = load_model('LGBM')


st.title('Prototype App Prediksi kredit loan status Menggunakan Model LGBM🤖')

st.header('Masukkan Data untuk Prediksi')
st.caption('*klik pada side bar untuk penjelasan setiap kolom*')

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True
st.button("let's get started",on_click = clicked, args=(1,))

if st.session_state.clicked[1]:
# Fitur Numerik
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        loan_amnt = st.number_input('Loan Amount', value=0)
    with col2:    
        funded_amnt = st.number_input('Funded Amount', value=0)
    with col3:
        funded_amnt_inv = st.number_input('Funded Amount Inv', value=0)
    with col4:
        int_rate = st.number_input('Interest Rate', value=0.0)


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        installment = st.number_input('Installment', value=0.0)
    with col2:
        annual_inc = st.number_input('Annual Income', value=0.0)
    with col3:
        dti = st.number_input('DTI', value=0.0)
    with col4:
        delinq_2yrs = st.number_input('Delinquency 2 Years', value=0)


    col1, col2, col3, col4 = st.columns(4)  
    with col1:
        inq_last_6mths = st.number_input('Inquiries Last 6 Months', value=0)
    with col2:
        open_acc = st.number_input('Open Accounts', value=0)
    with col3:    
        pub_rec = st.number_input('Public Records', value=0)
    with col4:    
        revol_bal = st.number_input('Revolving Balance', value=0)

    col1, col2, col3, col4 = st.columns(4)  
    with col1:
        revol_util = st.number_input('Revolving Utilization', value=0.0)
    with col2:
        total_acc = st.number_input('Total Accounts', value=0)
    with col3:    
        out_prncp = st.number_input('Outstanding Principal', value=0.0)
    with col4:
        out_prncp_inv = st.number_input('Outstanding Principal Inv', value=0.0)

    col1, col2, col3, col4 = st.columns(4)  
    with col1:
        total_pymnt = st.number_input('Total Payment', value=0.0)
    with col2:
        total_pymnt_inv = st.number_input('Total Payment Inv', value=0.0)
    with col3:    
        total_rec_prncp = st.number_input('Total Received Principal', value=0.0)
    with col4:    
        total_rec_int = st.number_input('Total Received Interest', value=0.0)

    col1, col2, col3, col4 = st.columns(4)  
    with col1:
        total_rec_late_fee = st.number_input('Total Received Late Fee', value=0.0)
    with col2:    
        recoveries = st.number_input('Recoveries', value=0.0)
    with col3:    
        collection_recovery_fee = st.number_input('Collection Recovery Fee', value=0.0)
    with col4:    
        last_pymnt_amnt = st.number_input('Last Payment Amount', value=0.0)

    col1, col2 = st.columns(2)  
    with col1:
        tot_cur_bal = st.number_input('Total Current Balance', value=0)
    with col2:
        total_rev_hi_lim = st.number_input('Total Revolving High Limit', value=0)

    # Fitur Kategorik
    col1, col2, col3, col4 = st.columns(4)  
    with col1:
        term = st.selectbox('Term', ['36 months', '60 months'])
    with col2:
        grade = st.selectbox('Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    with col3:    
        emp_length = st.selectbox('Employment Length', ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    with col4:    
        home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])

    col1, col2, col3 = st.columns(3)  
    with col1:
        verification_status = st.selectbox('Verification Status', ['Verified', 'Source Verified', 'Not Verified'])
    with col2:
        purpose = st.selectbox('Purpose', ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business', 'car','medical','moving','vacation','wedding','house','educational','renewable_energy', 'other'])
    with col3:    
        initial_list_status = st.selectbox('Initial List Status', ['w', 'f'])

    # Fitur Datetime
    col1, col2, col3, col4 = st.columns(4)  
    with col1:
        earliest_cr_line = st.date_input('Earliest Credit Line(y/m/d)')
    with col2:    
        issue_d = st.date_input('Issue Date(y/m/d)')
    with col3:
        last_pymnt_d = st.date_input('Last Payment Date(y/m/d)')
    with col4:    
        last_credit_pull_d = st.date_input('Last Credit Pull Date(y/m/d)')

    # Buat dataframe dari input pengguna
    input_data = {
        'loan_amnt': loan_amnt,
        'funded_amnt': funded_amnt,
        'funded_amnt_inv': funded_amnt_inv,
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'out_prncp': out_prncp,
        'out_prncp_inv': out_prncp_inv,
        'total_pymnt': total_pymnt,
        'total_pymnt_inv': total_pymnt_inv,
        'total_rec_prncp': total_rec_prncp,
        'total_rec_int': total_rec_int,
        'total_rec_late_fee': total_rec_late_fee,
        'recoveries': recoveries,
        'collection_recovery_fee': collection_recovery_fee,
        'last_pymnt_amnt': last_pymnt_amnt,
        'tot_cur_bal': tot_cur_bal,
        'total_rev_hi_lim': total_rev_hi_lim,
        'term': term,
        'grade': grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'verification_status': verification_status,
        'purpose': purpose,
        'initial_list_status': initial_list_status,
        'issue_d': issue_d,
        'last_pymnt_d': last_pymnt_d,
        'last_credit_pull_d': last_credit_pull_d,
        'earliest_cr_line': earliest_cr_line
    }

    # Konversi fitur datetime ke datetime64
    input_data['issue_d'] = pd.to_datetime(input_data['issue_d'], format='%Y-%m-%d')
    input_data['last_pymnt_d'] = pd.to_datetime(input_data['last_pymnt_d'], format='%Y-%m-%d')
    input_data['last_credit_pull_d'] = pd.to_datetime(input_data['last_credit_pull_d'], format='%Y-%m-%d')
    input_data['earliest_cr_line'] = pd.to_datetime(input_data['earliest_cr_line'], format='%Y-%m-%d')


    input_df = pd.DataFrame([input_data])

    if st.button('Prediksi'):
        # Lakukan prediksi menggunakan model
        prediction_result = predict_model(model, data=input_df)

        # Asumsikan bahwa hasil prediksi ada di kolom 'Label' (ubah sesuai dengan output actual dari predict_model)
        prediction = prediction_result.iloc[0]['prediction_label']

        st.write('prediction result:')
        st.write(prediction_result)

        if prediction == 'good':
            st.success(f"loan status: {prediction.capitalize()}")
        else:
            st.error(f"loan status: {prediction.capitalize()}")




with st.sidebar:
    with st.expander('About this app'):
        st.markdown ("app ini adalah prototype yang dibuat sebagai latihan, app ini dapat membantu pemberi hutang dalam mendiagnosa peminjam apakah nantinya dia akan membayar pinjamannya tau tidak sehingga pemberi hutang tidak mengalami kerugian")
    with st.expander('feature dictionary'):
        st.markdown("""
        **1. loan_amnt**: Jumlah pinjaman yang diajukan oleh peminjam.

        **2. funded_amnt**: Jumlah pinjaman yang disetujui dan didanai oleh pemberi pinjaman.

        **3. funded_amnt_inv**: Jumlah pinjaman yang didanai oleh investor individu.

        **4. int_rate**: Suku bunga tahunan dari pinjaman, dinyatakan dalam persentase.

        **5. installment**: Jumlah pembayaran bulanan yang harus dibayarkan oleh peminjam.

        **6. annual_inc**: Pendapatan tahunan peminjam.

        **7. dti**: Debt-to-Income Ratio, rasio antara total utang bulanan dan pendapatan bulanan peminjam.

        **8. delinq_2yrs**: Jumlah kali peminjam mengalami tunggakan pembayaran dalam dua tahun terakhir.

        **9. inq_last_6mths**: Jumlah permintaan kredit (credit inquiries) dalam enam bulan terakhir.

        **10. open_acc**: Jumlah akun kredit yang masih terbuka.

        **11. pub_rec**: Jumlah catatan publik negatif (misalnya kebangkrutan) yang terdaftar.

        **12. revol_bal**: Saldo bergulir pada akun kredit peminjam.

        **13. revol_util**: Tingkat penggunaan kredit bergulir, dinyatakan sebagai persentase dari total kredit yang tersedia.

        **14. total_acc**: Total jumlah akun kredit yang dimiliki peminjam.

        **15. out_prncp**: Sisa pokok pinjaman yang belum dibayar.

        **16. out_prncp_inv**: Sisa pokok pinjaman yang belum dibayar oleh investor individu.

        **17. total_pymnt**: Total jumlah pembayaran yang telah dilakukan oleh peminjam.

        **18. total_pymnt_inv**: Total jumlah pembayaran yang telah diterima oleh investor individu.

        **19. total_rec_prncp**: Total jumlah pokok pinjaman yang telah dibayar kembali oleh peminjam.

        **20. total_rec_int**: Total jumlah bunga yang telah dibayar oleh peminjam.

        **21. total_rec_late_fee**: Total jumlah denda keterlambatan yang telah dibayar oleh peminjam.

        **22. recoveries**: Jumlah dana yang berhasil dipulihkan dari pinjaman yang gagal bayar.

        **23. collection_recovery_fee**: Biaya yang dikeluarkan untuk proses pemulihan koleksi.

        **24. last_pymnt_amnt**: Jumlah pembayaran terakhir yang dilakukan oleh peminjam.

        **25. tot_cur_bal**: Total saldo kredit saat ini yang dimiliki oleh peminjam.

        **26. total_rev_hi_lim**: Total batas kredit tertinggi yang pernah diberikan kepada peminjam.

        **27. term**: Jangka waktu pinjaman.

        **28. grade**: Klasifikasi risiko kredit yang diberikan kepada peminjam.

        **29. emp_length**: Lama bekerja peminjam di pekerjaan saat ini.

        **30. home_ownership**: Status kepemilikan rumah peminjam.

        **31. verification_status**: Status verifikasi pendapatan peminjam.

        **32. purpose**: Tujuan dari pinjaman.

        **33. initial_list_status**: Status awal dari daftar pinjaman.

        **34. earliest_cr_line**: Tanggal pembukaan akun kredit pertama peminjam.
                    
        **35. issue_d**: Tanggal saat pinjaman diberikan.

        **36. last_pymnt_d**: Tanggal pembayaran terakhir yang dilakukan oleh peminjam.

        **37. last_credit_pull_d**: Tanggal terakhir informasi kredit peminjam ditarik.

        """)
    with st.expander('Documentation'):
        st.markdown ("https://github.com/Shodiqfathoni/id-x-Partners-data-scientist-credit-risk-project-/blob/main/idx_partners.ipynb")

        
    st.divider()

    st.caption("<p style ='text-align:center'> shodiq </p>",unsafe_allow_html = True )

