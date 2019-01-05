#/bin/bash

./bfs -f $1 -s > $2"-bfs.csv"
./bc -f $1 -s > $2"-bc.csv"
./cc -f $1 -s > $2"-cc.csv"
./tc -f $1 -s > $2"-tc.csv"
./pr -f $1 -s > $2"-pr.csv"

