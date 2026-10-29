# Apply a patch to a source tree once (skipped when it is already applied).
#   cmake -DSRC=<source dir> -DPATCH=<patch file> -P apply_patch.cmake
execute_process(COMMAND git apply --reverse --check "${PATCH}"
                WORKING_DIRECTORY "${SRC}" RESULT_VARIABLE already_applied OUTPUT_QUIET ERROR_QUIET)
if(NOT already_applied EQUAL 0)
    execute_process(COMMAND git apply "${PATCH}" WORKING_DIRECTORY "${SRC}" RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "Could not apply ${PATCH} to ${SRC} (is ggml at the pinned commit?)")
    endif()
    message(STATUS "Applied ${PATCH}")
endif()
