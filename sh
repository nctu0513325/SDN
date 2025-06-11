for web in $WEBS; do for ip in $IPS; do echo "TEST $web -> $ip" &&  sudo nsenter -t `docker inspect -f '{{ .State.Pid }}' web650$web` -n mtr -c 3 -r $ip; done; done
