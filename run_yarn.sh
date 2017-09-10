#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../deps/lib:/usr/hdp/2.5.3.0-37/usr/lib/
export CLASSPATH=$CLASSPATH:`hadoop classpath --glob`
export HADOOP_HDFS_HOME='/usr/hdp'
export HADOOP_YARN_HOME=$HADOOP_HOME
export HADOOP_ROOT_LOGGER=DEBUG,console

train_data="hdfs://100.94.4.100/mining4/lda_input/part*"
model_out="hdfs://100.94.4.100/mining4/lda_model_out/"
iter_num=100
nworker=80
nserver=20

hadoop fs -rmr $model_out
cat ./guide/demo_hdfs.conf > ./lda.conf
echo "model_out=\"$model_out\"" >> ./lda.conf
echo "train_data=\"$train_data\"" >> ./lda.conf
echo "max_data_pass=$iter_num" >> ./lda.conf

python /data/wbsvr/services/NagaRanking/tracker/dmlc_yarn.py --log-level DEBUG --vcores 2 -mem 600 -n $nworker -s $nserver -q queue2 build/lda.dmlc ./lda.conf

# dump part of word topic
build/dump.dmlc model_in="$model_out/word_topic/_part-0" dump_out="data/dump_0.txt" need_inverse=1
