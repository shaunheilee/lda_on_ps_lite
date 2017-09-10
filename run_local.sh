#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../deps/lib:$HADOOP_HOME/lib/native
export CLASSPATH=$CLASSPATH:`hadoop classpath --glob`
#hadoop fs -rmr /mining4/lda/test/*
#hadoop fs -cp /mining4/lda/input/part-$1 /mining4/lda/test/
#hadoop fs -cp /mining4/lda/input/part-$2 /mining4/lda/test/
python ../../dmlc-core/tracker/dmlc_local.py -n 1 -s 4 build/lda.dmlc guide/demo.conf

for((i=0;i<4;i++))
do
build/dump.dmlc model_in="hdfs://100.94.4.100/mining4/lda/model_out/word_topic/_part-$i" dump_out="data/dump_$i.txt" need_inverse=1
done
