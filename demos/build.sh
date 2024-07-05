declare -a demo_lists=("yolov8" "yolov8_pose" "mbf_arcface")

BUILD_DIR="$(pwd)/build"
INSTALL_DIR="$(pwd)/install"
ENABLE_ASAN=OFF

#Replace this with your own cross-compilation toolchain
GCC_COMPILER=~/ws/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf

export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

show_demo_menu() {
    echo "Please select a demo to build:"
    for ((i = 0; i < ${#demo_lists[@]}; i++)); do
        echo "$((i + 1))) ${demo_lists[$i]}"
    done
}

execute() {
    BUILD_DIR="${BUILD_DIR}/${demo_lists[$1 - 1]}"
    INSTALL_DIR="${INSTALL_DIR}/${demo_lists[$1 - 1]}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}" || exit
    cmake "../../${demo_lists[$1 - 1]}" \
        -DTARGET_SOC=rv1106 \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=armhf \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_ASAN=${ENABLE_ASAN} \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    make -j4
    make install
}

show_demo_menu
read -r demo
execute "$demo"
