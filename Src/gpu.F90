module gpumodule

   use cudafor
   use mol_cuda
      implicit none

      integer(4),device :: kn_d




      contains

attributes(global) subroutine DUTwoBodyEwaldRecStd_cuda
   !character(40), parameter :: txroutine ='UEwaldRecStd'
   implicit none
   integer(4) :: kn, nx, ny, nz, ia, ialoc, ikvec2
   real(8)    :: term, termnew, termold

!   if (ltime) call CpuAdd('start', txroutine, 3, uout)

! ... calculate eikxtm, eikytm, and eikztm for moving particles

   call EwaldSetArrayTM_cuda

   !kn = kvecoffmyid_d
   kn = 0
   ikvec2 = 0
   do nz = 0, ncut_d
      do ny = 0, ncut_d
         if (ny**2+nz**2 > ncut2_d) cycle
         ikvec2 = ikvec2+1
         !print *, "kvec1: ", kvecmyid_d(1)
         !print *, "kvec2: ", kvecmyid_d(2)
         !if (ikvec2 < kvecmyid_d(1) .or. ikvec2 > kvecmyid_d(2)) cycle  ! parallelize over k-vectors
         do ialoc = 1, natm_d
            ia = ianatm_d(ialoc)
            eikyzm_d(ia)      = conjg(eiky_d(ia,ny))     *eikz_d(ia,nz)
            eikyzp_d(ia)      =       eiky_d(ia,ny)      *eikz_d(ia,nz)
            eikyzmtm_d(ialoc) = conjg(eikytm_d(ialoc,ny))*eikztm_d(ialoc,nz)
            eikyzptm_d(ialoc) =       eikytm_d(ialoc,ny) *eikztm_d(ialoc,nz)
         end do

         do nx = 0, ncut_d
            if ((lbcrd_d .or. lbcto_d) .and. (mod((nx+ny+nz),2) /= 0)) cycle      ! only even nx+ny+nz for RD and TO bc
            if (nx**2+ny**2+nz**2 > ncut2_d) cycle
            if (nx == 0 .and. ny == 0 .and. nz == 0) cycle
            kn = kn + 1
            print *, kn, nx, ny, nz
            sumeikrtm_d(kn,1) = sumeikr_d(kn,1)
            sumeikrtm_d(kn,2) = sumeikr_d(kn,2)
            sumeikrtm_d(kn,3) = sumeikr_d(kn,3)
            sumeikrtm_d(kn,4) = sumeikr_d(kn,4)
            do ialoc = 1, natm_d
               ia = ianatm_d(ialoc)
               sumeikrtm_d(kn,1) = sumeikrtm_d(kn,1)+az_d(ia)*  &
                  (conjg(eikxtm_d(ialoc,nx))*eikyzmtm_d(ialoc) - conjg(eikx_d(ia,nx))*eikyzm_d(ia))
               sumeikrtm_d(kn,2) = sumeikrtm_d(kn,2)+az_d(ia)*  &
                  (conjg(eikxtm_d(ialoc,nx))*eikyzptm_d(ialoc) - conjg(eikx_d(ia,nx))*eikyzp_d(ia))
               sumeikrtm_d(kn,3) = sumeikrtm_d(kn,3)+az_d(ia)*  &
                        (eikxtm_d(ialoc,nx) *eikyzmtm_d(ialoc) -       eikx_d(ia,nx) *eikyzm_d(ia))
               sumeikrtm_d(kn,4) = sumeikrtm_d(kn,4)+az_d(ia)*  &
                        (eikxtm_d(ialoc,nx) *eikyzptm_d(ialoc) -       eikx_d(ia,nx) *eikyzp_d(ia))
            end do

            termnew = real(sumeikrtm_d(kn,1))**2 + aimag(sumeikrtm_d(kn,1))**2 + real(sumeikrtm_d(kn,2))**2 + aimag(sumeikrtm_d(kn,2))**2 &
                    + real(sumeikrtm_d(kn,3))**2 + aimag(sumeikrtm_d(kn,3))**2 + real(sumeikrtm_d(kn,4))**2 + aimag(sumeikrtm_d(kn,4))**2
            termold = real(sumeikr_d(kn,1))**2   + aimag(sumeikr_d(kn,1))**2   + real(sumeikr_d(kn,2))**2   + aimag(sumeikr_d(kn,2))**2 &
                    + real(sumeikr_d(kn,3))**2   + aimag(sumeikr_d(kn,3))**2   + real(sumeikr_d(kn,4))**2   + aimag(sumeikr_d(kn,4))**2
            term    = kfac_d(kn)*(termnew - termold)
            durec_d   = durec_d + term

         end do
      end do
   end do
   !if (ltime) call CpuAdd('stop', txroutine, 3, uout)

end subroutine DUTwoBodyEwaldRecStd_cuda

attributes(grid_global) subroutine DUTwoBodyEwaldRecStd_cuda_mod
   !character(40), parameter :: txroutine ='UEwaldRecStd'
   use cooperative_groups
   integer(4) :: kn, nx, ny, nz, ia, ialoc, istat, id
   real(8)    :: term, termnew, termold
   type(grid_group) :: gg

!   if (ltime) call CpuAdd('start', txroutine, 3, uout)

! ... calculate eikxtm, eikytm, and eikztm for moving particles

   gg = this_grid()
   if (threadIDx%x == 1) then
      call EwaldSetArrayTM_cuda
   end if
   call syncthreads(gg)

   id = (blockIDx%x - 1)*blockDim%x + threadIDx%x
   nx = mod(id-1,ncut_d+1)
   ny = mod(floor(real(id-1)/(ncut_d+1)),ncut_d+1)
   nz = floor(real(id-1)/((ncut_d+1)**2))
   kn = knid_d(id)
   !ny = mod(threadIDx%x-1,ncut_d+1)
   !nz = mod(floor(real(threadIDx%x-1)/(ncut_d+1)),ncut_d+1)
   !id = threadIDx%x

         if (ny**2+nz**2 > ncut2_d) goto 400
         do ialoc = 1, natm_d
            ia = ianatm_d(ialoc)
            eikyzm2_d(kn,ia)      = conjg(eiky_d(ia,ny))     *eikz_d(ia,nz)
            eikyzp2_d(kn,ia)      =       eiky_d(ia,ny)      *eikz_d(ia,nz)
            eikyzmtm2_d(kn,ialoc) = conjg(eikytm_d(ialoc,ny))*eikztm_d(ialoc,nz)
            eikyzptm2_d(kn,ialoc) =       eikytm_d(ialoc,ny) *eikztm_d(ialoc,nz)
         end do

         !do nx = 0, ncut_d
            if ((lbcrd_d .or. lbcto_d) .and. (mod((nx+ny+nz),2) /= 0)) goto 400      ! only even nx+ny+nz for RD and TO bc
            if (nx**2+ny**2+nz**2 > ncut2_d) goto 400
            if (nx == 0 .and. ny == 0 .and. nz == 0) goto 400
            !istat = atomicAdd(kn_d,1)
            !kn = kn_d
            sumeikrtm_d(kn,1) = sumeikr_d(kn,1)
            sumeikrtm_d(kn,2) = sumeikr_d(kn,2)
            sumeikrtm_d(kn,3) = sumeikr_d(kn,3)
            sumeikrtm_d(kn,4) = sumeikr_d(kn,4)
            do ialoc = 1, natm_d
               ia = ianatm_d(ialoc)
               sumeikrtm_d(kn,1) = sumeikrtm_d(kn,1)+az_d(ia)*  &
                  (conjg(eikxtm_d(ialoc,nx))*eikyzmtm2_d(kn,ialoc) - conjg(eikx_d(ia,nx))*eikyzm2_d(kn,ia))
               sumeikrtm_d(kn,2) = sumeikrtm_d(kn,2)+az_d(ia)*  &
                  (conjg(eikxtm_d(ialoc,nx))*eikyzptm2_d(kn,ialoc) - conjg(eikx_d(ia,nx))*eikyzp2_d(kn,ia))
               sumeikrtm_d(kn,3) = sumeikrtm_d(kn,3)+az_d(ia)*  &
                        (eikxtm_d(ialoc,nx) *eikyzmtm2_d(kn,ialoc) -       eikx_d(ia,nx) *eikyzm2_d(kn,ia))
               sumeikrtm_d(kn,4) = sumeikrtm_d(kn,4)+az_d(ia)*  &
                        (eikxtm_d(ialoc,nx) *eikyzptm2_d(kn,ialoc) -       eikx_d(ia,nx) *eikyzp2_d(kn,ia))
            end do

            termnew = real(sumeikrtm_d(kn,1))**2 + aimag(sumeikrtm_d(kn,1))**2 + real(sumeikrtm_d(kn,2))**2 + aimag(sumeikrtm_d(kn,2))**2 &
                    + real(sumeikrtm_d(kn,3))**2 + aimag(sumeikrtm_d(kn,3))**2 + real(sumeikrtm_d(kn,4))**2 + aimag(sumeikrtm_d(kn,4))**2
            termold = real(sumeikr_d(kn,1))**2   + aimag(sumeikr_d(kn,1))**2   + real(sumeikr_d(kn,2))**2   + aimag(sumeikr_d(kn,2))**2 &
                    + real(sumeikr_d(kn,3))**2   + aimag(sumeikr_d(kn,3))**2   + real(sumeikr_d(kn,4))**2   + aimag(sumeikr_d(kn,4))**2
            term    = kfac_d(kn)*(termnew - termold)
            !durec_d   = durec_d + term
            istat = atomicAdd(durec_d, term)
         !end do

   400 continue
   !if (ltime) call CpuAdd('stop', txroutine, 3, uout)

end subroutine DUTwoBodyEwaldRecStd_cuda_mod


attributes(device) subroutine EwaldSetArrayTM_cuda

   use EnergyModule
   use mol_cuda
   implicit none

   integer(4) :: ialoc, icut

   do ialoc = 1, natm_d
      eikxtm_d(ialoc,0) = cmplx(One_d,Zero_d)
      eikytm_d(ialoc,0) = cmplx(One_d,Zero_d)
      eikztm_d(ialoc,0) = cmplx(One_d,Zero_d)
      eikxtm_d(ialoc,1) = cmplx(cos(TwoPiBoxi_d(1)*rtm_d(1,ialoc)),sin(TwoPiBoxi_d(1)*rtm_d(1,ialoc)))
      eikytm_d(ialoc,1) = cmplx(cos(TwoPiBoxi_d(2)*rtm_d(2,ialoc)),sin(TwoPiBoxi_d(2)*rtm_d(2,ialoc)))
      eikztm_d(ialoc,1) = cmplx(cos(TwoPiBoxi_d(3)*rtm_d(3,ialoc)),sin(TwoPiBoxi_d(3)*rtm_d(3,ialoc)))
   end do
   do icut = 2, ncut_d
      do ialoc = 1, natm_d
         eikxtm_d(ialoc,icut) = eikxtm_d(ialoc,icut-1)*eikxtm_d(ialoc,1)
         eikytm_d(ialoc,icut) = eikytm_d(ialoc,icut-1)*eikytm_d(ialoc,1)
         eikztm_d(ialoc,icut) = eikztm_d(ialoc,icut-1)*eikztm_d(ialoc,1)
      end do
   end do

end subroutine EwaldSetArrayTM_cuda

attributes(global) subroutine EwaldUpdateArray_cuda

   use EnergyModule
   implicit none

   integer(4) :: ia, ialoc, icut

   !if (txewaldrec == 'std') then
      do icut = 0, ncut_d
         do ialoc = 1, natm_d
            ia = ianatm_d(ialoc)
            eikx_d(ia,icut) = eikxtm_d(ialoc,icut)
            eiky_d(ia,icut) = eikytm_d(ialoc,icut)
            eikz_d(ia,icut) = eikztm_d(ialoc,icut)
         end do
      end do
      sumeikr_d(1:nkvec_d,1:4) = sumeikrtm_d(1:nkvec_d,1:4)
   !end if

end subroutine EwaldUpdateArray_cuda

attributes(global) subroutine DUTwoBodyEwaldSurf_cuda
   integer(4) :: ia, ialoc
   real(8)    :: fac, term, sumqrx, sumqry, sumqrz, sumqrxt, sumqryt, sumqrzt
   real(8)    :: Three=3.0d0

   if (.not.lewald2dlc_d) then

      fac = TwoPi_d/(Three*vol_d)
      sumqrx = sum(az_d(1:na_d)*r_d(1,1:na_d))
      sumqry = sum(az_d(1:na_d)*r_d(2,1:na_d))
      sumqrz = sum(az_d(1:na_d)*r_d(3,1:na_d))
      sumqrxt = sumqrx
      sumqryt = sumqry
      sumqrzt = sumqrz
      do ialoc = 1, natm_d
         ia = ianatm_d(ialoc)
         sumqrxt = sumqrxt + az_d(ia)*(rtm_d(1,ialoc)-r_d(1,ia))
         sumqryt = sumqryt + az_d(ia)*(rtm_d(2,ialoc)-r_d(2,ia))
         sumqrzt = sumqrzt + az_d(ia)*(rtm_d(3,ialoc)-r_d(3,ia))
      end do
      term = fac*((sumqrxt**2+sumqryt**2+sumqrzt**2) - (sumqrx**2+sumqry**2+sumqrz**2))
      durec_d = durec_d + term

   else

      fac = TwoPi_d/vol_d
      sumqrz = sum(az_d(1:na_d)*r_d(3,1:na_d))
      sumqrzt = sumqrz
      do ialoc = 1, natm_d
         ia = ianatm_d(ialoc)
         sumqrzt = sumqrzt + az_d(ia)*(rtm_d(3,ialoc)-r_d(3,ia))
      end do
      term = fac*(sumqrzt**2 - sumqrz**2)
      durec_d = durec_d + term

   end if
end subroutine DUTwoBodyEwaldSurf_cuda

end module gpumodule
