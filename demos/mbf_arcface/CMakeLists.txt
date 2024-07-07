cmake_minimum_required(VERSION 3.10)

project(rknn_mbf_arcface_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(rknpu_arcface_file ${CMAKE_CURRENT_SOURCE_DIR}/src/mbf_arcface.cc)
add_definitions(-DRV1106_1103)
#dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/allocator/dma)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess.cc
    ${rknpu_arcface_file}
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
)

add_executable(eval
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eval.cc
    ${rknpu_arcface_file}
)

target_link_libraries(eval
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
    log
)
endif()

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LIBRKNNRT_INCLUDES}
)

target_include_directories(eval PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} eval DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/lecun1.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/lecun2.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/lecun3.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/altman1.jpg DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
file(GLOB EVAL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/eval_dataset/yale_face/*.jpg")
install(FILES ${EVAL_FILES} DESTINATION eval_dataset/yale_face)