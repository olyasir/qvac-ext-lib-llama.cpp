set(TARGET fabric-ocr)
add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
target_compile_features(${TARGET} PRIVATE cxx_std_20)
target_include_directories(${TARGET} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/fabric
)
