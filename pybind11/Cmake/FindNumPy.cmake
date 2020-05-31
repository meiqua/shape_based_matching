# - Find numpy
# Find the native numpy includes
# This module defines
#  PYTHON_NUMPY_INCLUDE_DIR, where to find numpy/arrayobject.h, etc.
#  PYTHON_NUMPY_FOUND, If false, do not try to use numpy headers.

if (PYTHON_NUMPY_INCLUDE_DIR)
  # in cache already
  set (PYTHON_NUMPY_FIND_QUIETLY TRUE)
endif (PYTHON_NUMPY_INCLUDE_DIR)

INCLUDE(FindPythonInterp)

IF(PYTHON_EXECUTABLE)
    EXEC_PROGRAM ("${PYTHON_EXECUTABLE}"
      ARGS "-c 'import numpy; print(numpy.get_include())'"
      OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
      RETURN_VALUE PYTHON_NUMPY_NOT_FOUND)

    if (PYTHON_NUMPY_INCLUDE_DIR)
      set (PYTHON_NUMPY_FOUND TRUE)
      set (PYTHON_NUMPY_INCLUDE_DIR ${PYTHON_NUMPY_INCLUDE_DIR} CACHE STRING "Numpy include path")
    else (PYTHON_NUMPY_INCLUDE_DIR)
      set(PYTHON_NUMPY_FOUND FALSE)
    endif (PYTHON_NUMPY_INCLUDE_DIR)
ENDIF(PYTHON_EXECUTABLE)

if (PYTHON_NUMPY_FOUND)
  if (NOT PYTHON_NUMPY_FIND_QUIETLY)
    message (STATUS "Numpy headers found")
  endif (NOT PYTHON_NUMPY_FIND_QUIETLY)
else (PYTHON_NUMPY_FOUND)
  if (PYTHON_NUMPY_FIND_REQUIRED)
    message (FATAL_ERROR "Numpy headers missing")
  endif (PYTHON_NUMPY_FIND_REQUIRED)
endif (PYTHON_NUMPY_FOUND)

MARK_AS_ADVANCED (PYTHON_NUMPY_INCLUDE_DIR)

