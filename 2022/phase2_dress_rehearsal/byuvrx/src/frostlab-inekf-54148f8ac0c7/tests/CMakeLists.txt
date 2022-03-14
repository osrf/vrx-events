#################################
set(GMOCK_LIBRARIES gmock gtest gtest_main)

# Add and Install Tests
function(add_and_install_tests TESTS TEST_LIBS)
  foreach(source_file ${TESTS})
    # message(STATUS "Making ${source_file}")
    get_filename_component(exe_suffix ${source_file} NAME_WE)
    string(REGEX REPLACE "_" "-" exe_suffix ${exe_suffix}) # underscore
    # message(STATUS "exe_suffix: ${exe_suffix}")
    set(EXE_NAME ${exe_suffix})

    add_executable(${EXE_NAME} ${source_file})

    # Link GMock
    target_link_libraries(${EXE_NAME} ${GMOCK_LIBRARIES})

    #Link Dependencies
    target_link_libraries(${EXE_NAME} ${TEST_LIBS})

    target_include_directories(${EXE_NAME} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      )  
    
    add_test(NAME ${EXE_NAME} COMMAND $<TARGET_FILE:${EXE_NAME}>)

    install(TARGETS ${EXE_NAME} EXPORT ${EXE_NAME}
      DESTINATION tests)

  endforeach(source_file)
endfunction()
#################################

add_subdirectory(Core)
add_subdirectory(SE2Models)
add_subdirectory(Inertial)