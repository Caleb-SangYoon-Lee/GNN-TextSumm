#
# AI-Hub에서 다운로드 받은 텍스트 길이 요약 데이터를 보여줌
#
#
# 테스트 데이터(총 40,132건)에 대한 통계는 아래와 같음
#
#    max_token_in_text     : 2048
#    max_token_in_sent     : 559
#    max_token_in_summ     : 399
#    n_long_article        : 113
#    n_null_article_skipped: 0
#
#    sent_length
#    -------------------
#    count  604519.000000
#    mean       35.047797
#    std        20.400267
#    min         1.000000
#    25%        21.000000
#    50%        31.000000
#    75%        45.000000
#    max       559.000000
#
# -----------------------------------------------------
#
# 학습용 데이터(총 321,052건)에 대한 통계는 아래와 같음
#
#     max_token_in_text:7830
#     max_token_in_sent:699
#     max_token_in_summ:576
#     n_long_article:899
#     n_null_article_skipped:0
#
#    sent_length
#    -------------------
#    count  4.825387e+06
#    mean   3.509082e+01
#    std    2.041120e+01
#    min    1.000000e+00
#    25%    2.100000e+01
#    50%    3.100000e+01
#    75%    4.500000e+01
#    max    6.990000e+02

python prep-data-00-eda.py --config-dir ./config --config-file config-base-v001.json --mode train
python prep-data-00-eda.py --config-dir ./config --config-file config-base-v001.json --mode test
