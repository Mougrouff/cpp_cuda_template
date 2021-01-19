set(EXECUTABLES
  "test_template"
)

set(LIB_NAME ${PROJECT_NAME}_lib)

foreach(exec ${EXECUTABLES})
  add_executable(${exec} ${exec}.cpp)
  target_link_libraries(${exec} ${LIB_NAME})

  if (MSVC)
    target_compile_options(${exec} PRIVATE "/IW4")
    target_compile_options(${exec} PRIVATE "/WX")
  else()
    target_compile_options(${exec} PRIVATE "-Wall")
    target_compile_options(${exec} PRIVATE "-Wextra")
    target_compile_options(${exec} PRIVATE "-Werror")
    target_compile_options(${exec} PRIVATE "-pedantic")
  endif()
endforeach()
