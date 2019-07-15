module precision_m

  implicit none
integer, parameter :: singlePrecision = kind(0.0)
integer, parameter :: doublePrecision = kind(0.0d0)

#ifdef DOUBLE
  integer, parameter :: fp_kind = doublePrecision
#else
  integer, parameter :: fp_kind = singlePrecision
#endif

end module precision_m
