#python main.py -x 0.003 -i 1 -C 0.5 -m Upwind &
#python main.py -x 0.003 -i 1 -C 0.5 -m limiter &
#python main2.py -x 0.1 -i 1 -C 0.57 -t 0.5,0.8,1.1,1.4 -m Upwind,Minmod &
#python main2.py -x 0.03 -i 2 -C 0.4 -t 0.5,0.6,0.7,0.8 -m Upwind,Minmod &
#python main2.py -x 0.03 -i 2 -C 0.5 -t 0.5,0.6,0.7,0.8 -m Upwind,Minmod &
#python main2.py -x 0.03 -i 2 -C 0.6 -t 0.5,0.6,0.7,0.8 -m Upwind,Minmod
python main2.py -x 0.003 -i 2 -C 0.1 -t 0.5,0.75,1,1.25 -m Upwind,Minmod
