module mol_cuda

    use Molmodule
    use cudafor
    implicit none


   real(8),    constant :: ThreeHalf_d = 1.5d0
   real(8),    constant :: SqTwo_d       = sqrt(Two)
   logical,device       :: lbcbox_d                 ! box-like cell (rätblock)
   logical,device       :: lbcrd_d                  ! rhombic dodecahedral cell
   logical,device       :: lbcto_d                  ! truncated octahedral cell
   real(8),device       :: boxlen_d(3)
   real(8),device       :: boxlen2_d(3)             ! boxlen/2
   logical,device       :: lPBC_d                   ! periodic boundary conditions
   real(8),device       :: dpbc_d(3)                ! =boxlen for some pbc, otherwise zero
   integer(4),device              :: np_d           ! number of particles
   integer(4), device    :: nptpt_d                  ! number of different particle type pairs
   logical,device                    :: lmonoatom_d
   real(8), device, allocatable       :: r2atat_d(:)     !
   integer(4), device, allocatable :: iptpn_d(:)     ! particle (1:np)               -> its particle type (1:npt)

   integer(4),device, allocatable :: iptpt_d(:,:)   ! two particle types (1:npt)    -> particle type pair (1:nptpt)

   logical,device       :: lmc_d                    ! flag for monte carlo simulation
   real(8),device       :: virial_d                 ! virial
   real(8), device, allocatable :: ro_d(:,:)         ! particle position
   integer(4),device, allocatable :: nneighpn_d(:)  ! particle (local) -> number of neighbours
   integer(4),device, allocatable :: jpnlist_d(:,:) ! ineigh (local list) and ip (global or local) -> neigbour particle (1:np)
   integer(4),device              :: nbuf_d         ! length of buffer
   real(8), device,    allocatable :: ubuf_d(:)      ! buffer for potential table
   real(8), device                 :: rcut2_d        ! rcut**2
   real(8), device,    allocatable :: r2umin_d(:)    ! lower limit squared of potential table
   integer(4),device, allocatable :: iubuflow_d(:)  ! points on the first entry for iatjat
   real(8), device,allocatable :: utwob_d(:)
   real(8), device  :: utot_d

!... in DuTotal
   logical, device      :: lhsoverlap_d
   integer(4),device    :: nptm_d
   integer(4), device, allocatable :: ipnptm_d(:)
   logical, device, allocatable    :: lptm_d(:)
   real(8), device, allocatable    :: rotm_d(:,:)
   logical, device                 :: lellipsoid_d
   logical, device                 :: lsuperball_d
   real(8), device, allocatable    :: dutwob_d(:)
   logical,device       :: lptmdutwob_d             ! flag for calulating dutobdy among moving particles
   real(8), device,allocatable :: utwobnew_d(:)
   real(8), device,allocatable :: utwobold_d(:)
   
   contains


subroutine AllocateDeviceParams


        use NListModule
        implicit none

        integer(4) :: istat

        allocate(iptpt_d(npt,npt))
        allocate(jpnlist_d(maxnneigh,np))
        allocate(utwob_d(0:nptpt))
        allocate(ro_d(3,np))
        allocate(r2umin_d(nptpt))
        allocate(r2atat_d(natat))
        allocate(iubuflow_d(nptpt))
        allocate(nneighpn_d(np))
        allocate(iptpn_d(np))
        allocate(ubuf_d(nbuf))
        allocate(rotm_d(3,np))
        allocate(lptm_d(np_alloc))
        allocate(ipnptm_d(np_alloc))
        allocate(dutwob_d(0:nptpt))
        allocate(utwobnew_d(0:nptpt))
        allocate(utwobold_d(0:nptpt))


end subroutine AllocateDeviceParams

subroutine TransferConstantParams

        implicit none
        
        integer(4) :: istat
        !istat = cudaMemcpy(boxlen2_d,boxlen2,3)
        !istat = cudaMemcpy(boxlen_d, boxlen,3)
        !istat = cudaMemcpy(dpbc_d, dpbc,3)
        !istat = cudaMemcpy(lPBC_d,lPBC,1) 
        !istat = cudaMemcpy(lbcbox_d,lbcbox,1) 
        !istat = cudaMemcpy(lbcrd_d,lbcrd,1) 
        !istat = cudaMemcpy(lbcto_d,lbcto,1) 
        !istat = cudaMemcpy(rcut2_d, rcut,1)
        !istat = cudaMemcpy(nptpt_d, nptpt,1)
        !istat = cudaMemcpy(r2atat_d,r2atat,natat) 
        !istat = cudaMemcpy(r2umin_d,r2umin,nptpt) 
        !istat = cudaMemcpy(iubuflow_d,iubuflow,nptpt) 
        !istat = cudaMemcpy(iptpn_d, iptpn, np)
        !istat = cudaMemcpy(iptpt_d,iptpt,nptpt)
        !istat = cudaMemcpy(nbuf_d, nbuf,1)
        !istat = cudaMemcpy(ubuf_d, ubuf, nbuf)
        !istat = cudaMemcpy(lmonoatom_d,lmonoatom,1)
        !istat = cudaMemcpy(lmc_d,lmc,1)
        !istat = cudaMemcpy(np_d,np,1)

        boxlen2_d = boxlen2
        boxlen_d = boxlen
        dpbc_d = dpbc
        lPBC_d = lPBC
        lbcbox_d = lbcbox
        lbcrd_d = lbcrd
        lbcto_d = lbcto
        rcut2_d = rcut2
        nptpt_d = nptpt
        r2atat_d = r2atat
        r2umin_d = r2umin
        iubuflow_d = iubuflow
        iptpn_d = iptpn
        iptpt_d = iptpt
        nbuf_d = nbuf
        ubuf_d = ubuf
        lmonoatom_d = lmonoatom
        lmc_d = lmc
        np_d = np
        lellipsoid_d = lellipsoid
        lsuperball_d = lsuperball
        lptmdutwob_d = lptmdutwob

end subroutine TransferConstantParams

subroutine TransferVarParamsToDevice

        use NListModule
        implicit none
        
        integer(4) :: istat
        !istat = cudaMemcpy2D(ro_d,ro,3*np)

        ro_d =ro
        utwob_d = u%twob
        virial_d = virial
        !istat = cudaMemcpy(nneighpn_d,nneighpn,np)
        nneighpn_d = nneighpn
        jpnlist_d = jpnlist
        utot_d = u%tot

        !istat = cudaMemcpy(nneighpn_d, nneighpn,np)
        !istat = cudaMemcpy(jpnlist_d,jpnlist,maxnneigh*np)

end subroutine TransferVarParamsToDevice

subroutine TransferVarParamsToHost

        implicit none

        integer(4) :: istat

        !istat = cudaMemcpy(ro_d,ro,3*np)
        !ro = ro_d
        virial = virial_d
        u%tot = utot_d
        u%twob = utwob_d
end subroutine TransferVarParamsToHost

subroutine TransferDUTotalVarToDevice

        use NListModule
        implicit none
        !logical, intent(in) :: lhsoverlap

        dutwob_d = du%twob
        nptm_d = nptm
        ipnptm_d = ipnptm
        nneighpn_d = nneighpn
        lptm_d = lptm
        rotm_d = rotm
        ro_d = ro

end subroutine TransferDUTotalVarToDevice

subroutine TransferDUTotalVarToHost

        implicit none
        !logical, intent(inout) :: lhsoverlap

        du%twob = dutwob_d


end subroutine TransferDUTotalVarToHost


!************************************************************************
!> \page PBCr2_cuda
!! **PBCr2_cuda**
!! *apply periodic boundary conditions and calculate r**2 on GPU*
!************************************************************************


attributes(device) subroutine PBCr2_cuda(dx,dy,dz,r2)!,boxlen,boxlen2,lPBC,lbcbox,&
                              !lbcrd,lbcto)

   implicit none

   real(8), intent(inout) :: dx, dy, dz
   real(8), intent(inout) :: r2
!   real(8), intent(inout)  :: boxlen, boxlen2
!   logical, intent(inout)  :: lPBC, lbcbox, lbcrd, lbcto
   !real(8)              :: Threehalf = 1.5d0
   !real(8)              :: SqTwo = sqrt(2.0d0)

   if (lPBC_d) then                                                              ! periodic boundary condition
      if (lbcbox_d) then                                                         ! box-like cell
         if (abs(dx) > boxlen2_d(1)) dx = dx - sign(dpbc_d(1),dx)
         if (abs(dy) > boxlen2_d(2)) dy = dy - sign(dpbc_d(2),dy)
         if (abs(dz) > boxlen2_d(3)) dz = dz - sign(dpbc_d(3),dz)
      else if (lbcrd_d) then                                                     ! rhombic dodecahedral cell
         if (abs(dx) > boxlen2_d(1)) dx = dx - sign(boxlen_d(1),dx)
         if (abs(dy) > boxlen2_d(2)) dy = dy - sign(boxlen_d(2),dy)
         if (abs(dz) > boxlen2_d(3)) dz = dz - sign(boxlen_d(3),dz)
         if (abs(dx) + abs(dy) + SqTwo_d*abs(dz) > boxlen_d(1)) then
            dx = dx - sign(boxlen2_d(1),dx)
            dy = dy - sign(boxlen2_d(2),dy)
            dz = dz - sign(boxlen2_d(3),dz)
         end if
      else if (lbcto_d) then                                                     ! truncated octahedral cell
         if (abs(dx) > boxlen2_d(1)) dx = dx - sign(boxlen_d(1),dx)
         if (abs(dy) > boxlen2_d(2)) dy = dy - sign(boxlen_d(2),dy)
         if (abs(dz) > boxlen2_d(3)) dz = dz - sign(boxlen_d(3),dz)
         if (abs(dx) + abs(dy) + abs(dz) > ThreeHalf_d*boxlen2_d(1)) then
            dx = dx - sign(boxlen2_d(1),dx)
            dy = dy - sign(boxlen2_d(2),dy)
            dz = dz - sign(boxlen2_d(3),dz)
         end if
      end if
   end if
   r2 = dx**2+dy**2+dz**2

end subroutine PBCr2_cuda





end module mol_cuda
