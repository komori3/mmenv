g++ -O2 -Wall -Wextra -std=c++17 $1 \
    -I/usr/local/include/opencv4 -L/usr/local/lib \
    -lopencv_core -lopencv_highgui -lopencv_imgproc