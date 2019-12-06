module mol_cuda

    use Molmodule
    use cudafor
    use precision_m
    implicit none
   logical :: lcuda = .true.

   real(fp_kind),    constant :: ThreeHalf_d = 1.5
   real(fp_kind),    constant :: SqTwo_d       = sqrt(Two)
   logical,device       :: lbcbox_d                 ! box-like cell (rätblock)
   logical,device       :: lbcrd_d                  ! rhombic dodecahedral cell
   logical,device       :: lbcto_d                  ! truncated octahedral cell
   real(fp_kind),device       :: boxlen_d(3)
   real(fp_kind),device       :: boxlen2_d(3)             ! boxlen/2
   real(fp_kind),device       :: boxleni_d(3)             ! boxlen/2
   real(fp_kind),device       :: dpbc_d(3)             ! /2
   logical,device       :: lPBC_d                   ! periodic boundary conditions
   integer(4),device              :: np_d           ! number of particles
   integer(4), device    :: nptpt_d                  ! number of different particle type pairs
   integer(4), device    :: npt_d
   logical,device                    :: lmonoatom_d
   real(fp_kind), device, allocatable       :: r2atat_d(:)     !
   integer(4), device, allocatable :: iptpn_d(:)     ! particle (1:np)               -> its particle type (1:npt)

   integer(4),device, allocatable :: iptpt_d(:,:)   ! two particle types (1:npt)    -> particle type pair (1:nptpt)

   logical,device       :: lmc_d                    ! flag for monte carlo simulation
   real(fp_kind),device       :: virial_d                 ! virial
   real(fp_kind), device, allocatable :: ro_d(:,:)         ! particle position
   integer(4),device, allocatable :: nneighpn_d(:)  ! particle (local) -> number of neighbours
   integer(4),device, allocatable :: jpnlist_d(:,:) ! ineigh (local list) and ip (global or local) -> neigbour particle (1:np)
   integer(4),device              :: nbuf_d         ! length of buffer
   real(fp_kind), device,    allocatable :: ubuf_d(:)      ! buffer for potential table
   real(fp_kind), device                 :: rcut2_d        ! rcut**2
   real(fp_kind), device,    allocatable :: r2umin_d(:)    ! lower limit squared of potential table
   integer(4),device, allocatable :: iubuflow_d(:)  ! points on the first entry for iatjat
   real(fp_kind), device,allocatable :: utwob_d(:)
   real(fp_kind), device  :: utot_d
   real(fp_kind), device  :: virtwob_d

!... potential
   real(fp_kind), device, allocatable :: ucoff_d(:)
   real(fp_kind), device :: scrlen_d

! MCPass
   integer(4),device           :: iptmove_d
   integer(4),device           :: ipmove_d
   real(fp_kind), device, allocatable :: dtran_d(:)
!... in DuTotal
   logical, device      :: lhsoverlap_d
   integer(4),device    :: nptm_d
   integer(4), device, allocatable :: ipnptm_d(:)
   logical, device, allocatable    :: lptm_d(:)
   real(fp_kind), device, allocatable    :: rotm_d(:,:)
   logical, device                 :: lellipsoid_d
   logical, device                 :: lsuperball_d
   real(fp_kind), device, allocatable    :: dutwob_d(:)
   logical,device       :: lptmdutwob_d             ! flag for calulating dutobdy among moving particles
   real(fp_kind), device,allocatable :: utwobnew_d(:)
   real(fp_kind), device,allocatable :: utwobold_d(:)
   real(fp_kind), allocatable :: dutwobold(:)


   integer(4) :: iinteractions
   integer(4),device :: iinteractions_d
   integer(4), device :: ierror_d
   integer(4), device :: sizeofblocks_d

   integer(4) :: threadssum
   integer(4),device :: threadssum_d

   ! bonds
   real(fp_kind), device, allocatable :: bond_d_k(:)
   real(fp_kind), device, allocatable :: bond_d_eq(:)
   real(fp_kind), device, allocatable :: bond_d_p(:)
   integer(4), device, allocatable :: bondnn_d(:,:)
   real(fp_kind) :: bond_aux
   logical, device :: lchain_d
   integer(4), device, allocatable :: ictpn_d(:)
   real(fp_kind), device, allocatable :: rsumrad(:,:)
   real(fp_kind), device :: clink_d_k
   real(fp_kind), device :: clink_d_eq
   real(fp_kind), device :: clink_d_p
   integer(4), device, allocatable :: bondcl_d(:,:)
   integer(4), device, allocatable :: nbondcl_d(:)
   logical, device :: lclink_d
   real(fp_kind), allocatable :: rsumrad_h(:,:)
   real(fp_kind), device, allocatable :: sig(:)
   real(fp_kind), device, allocatable :: eps(:)

   integer(4), allocatable :: seedsnp(:)

   real(fp_kind), device :: beta_d

   logical :: lseq
   logical :: lcuda_mcpass
   logical,device :: ltest_cuda


   real(8) :: u_aux
   integer(4), device :: iseed_d
      !for random generators
      real(8),device :: am_dev
      integer(k4b),device :: ix_dev=-1,iy_dev=-1
      integer(k4b),device :: ix_dev2=-1,iy_dev2=-1
      integer, parameter :: k4b_d=selected_int_kind(9) ! = 4 on intel fortran and gfortran
      real(8), device, allocatable :: am_d(:)
      integer(k4b),device, allocatable :: ix_d(:),iy_d(:)
      integer(4), allocatable :: seeds(:)
      integer(4),device, allocatable :: seeds_d(:)
      integer(4) :: icounter
      integer(4),device :: icounter_d
      integer(4) :: icounter2
      integer(4),device :: icounter2_d


   contains


subroutine AllocateDeviceParams


        !use NListModule
        use Random_Module
        implicit none

        integer(4) :: istat

   if(ltime) call CpuAdd('start', 'allocation', 1, uout)
        allocate(iptpt_d(npt,npt))
        !allocate(jpnlist_d(maxnneigh,npartperproc))
        allocate(utwob_d(0:nptpt))
        allocate(ro_d(3,np_alloc))
        allocate(r2umin_d(natat))
        allocate(r2atat_d(natat))
        allocate(iubuflow_d(natat))
        !allocate(nneighpn_d(np_alloc))
        write(*,*) "1"
        allocate(iptpn_d(np_alloc))
        allocate(ubuf_d(nbuf))
        allocate(rotm_d(3,np_alloc))
        allocate(lptm_d(np_alloc))
        allocate(ipnptm_d(np_alloc))
        allocate(dutwob_d(0:nptpt))
        allocate(utwobnew_d(0:nptpt))
        allocate(utwobold_d(0:nptpt))
        allocate(dutwobold(0:nptpt))
        allocate(ucoff_d(natat))
        write(*,*) "2"
        write(*,*) "1"
        allocate(seedsnp(np_alloc))
        write(*,*) "2"
        allocate(seeds_d(np_alloc))
        write(*,*) "3"
        allocate(bondnn_d(2,np))
        allocate(ictpn_d(np_alloc))
        allocate(bondcl_d(4,np_alloc))
        write(*,*) "4"
        allocate(rsumrad(npt,npt))
        write(*,*) "5"
        allocate(rsumrad_h(npt,npt))
        write(*,*) "6"
        allocate(dtran_d(npt))
        write(*,*) "7"
        if (lchain) then
           allocate(bond_d_k(nct))
           bond_d_k = 0.0
           allocate(bond_d_eq(nct))
           bond_d_eq = 0.0
           allocate(bond_d_p(nct))
           bond_d_p = 0
        end if
        allocate(nbondcl_d(np_alloc))
        allocate(ix_d(np_alloc))
        allocate(iy_d(np_alloc))
        allocate(am_d(np_alloc))
   if(ltime) call CpuAdd('stop', 'allocation', 1, uout)


end subroutine AllocateDeviceParams

subroutine TransferConstantParams

        use Molmodule
        use Random_Module
        use PotentialModule
        implicit none
        
        integer(4) :: istat, ipt, jpt, ict
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

   if(ltime) call CpuAdd('start', 'transferconstant', 1, uout)
        boxlen2_d = boxlen2
        boxlen_d = boxlen
        boxleni_d = boxleni
        dpbc_d = dpbc
        lPBC_d = lPBC
        lbcbox_d = lbcbox
        lbcrd_d = lbcrd
        lbcto_d = lbcto
        rcut2_d = rcut2
        scrlen_d = scrlen
        nptpt_d = nptpt
        !dtran_d = dtran
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
        npt_d = npt
        lellipsoid_d = lellipsoid
        lsuperball_d = lsuperball
        lptmdutwob_d = lptmdutwob
        iinteractions_d = iinteractions
        lchain_d = lchain
        ictpn_d = ictpn
        if (lchain) then
           do ict =1, nct
              bond_d_k(ict) = bond(ict)%k
              bond_d_eq(ict) = bond(ict)%eq
              bond_d_p(ict) = bond(ict)%p
           end do
           bondnn_d = bondnn
        end if

        lclink_d = lclink
   if(lclink) then
        clink_d_k = clink%k
        clink_d_eq = clink%eq
        clink_d_p = clink%p
        bondcl_d = bondcl
        nbondcl_d = nbondcl
   end if
        lcuda = .true.
        lseq = .true.
        lcuda_mcpass = .false.
        ltest_cuda = .true.

        ro_d = ro
        sizeofblocks_d = 512
        threadssum =16
        threadssum_d = threadssum

        seeds_d = seedsnp
        beta_d = beta
        do ipt =1, npt
           do jpt = 1, npt
              rsumrad_h(ipt,jpt) = radat(ipt) + radat(jpt)
              rsumrad_h(ipt,jpt) = rsumrad_h(ipt,jpt)**2
           end do
        end do
        do ipt = 1, natat
           ucoff_d(ipt) = ucoff(1,ipt)
        end do
        rsumrad = rsumrad_h
   if(ltime) call CpuAdd('stop', 'transferconstant', 1, uout)
        ix_dev = ix
        iy_dev = iy
        am_dev = am
        iseed_d = iseed

end subroutine TransferConstantParams

subroutine TransferVarParamsToDevice

        !use NListModule
        implicit none

        integer(4) :: istat
        !istat = cudaMemcpy2D(ro_d,ro,3*np)

   if(ltime) call CpuAdd('start', 'transferVartoDevice', 1, uout)
        ro_d =ro
        !utwob_d = u%twob
        utwob_d = 0.0
        virtwob_d = 0.0
        virial_d = virial
        !istat = cudaMemcpy(nneighpn_d,nneighpn,np)
        !nneighpn_d = nneighpn
        !jpnlist_d = jpnlist
        utot_d = u%tot
        virtwob_d = 0.0
   if(ltime) call CpuAdd('start', 'transferVartoDevice', 1, uout)

        !istat = cudaMemcpy(nneighpn_d, nneighpn,np)
        !istat = cudaMemcpy(jpnlist_d,jpnlist,maxnneigh*np)

end subroutine TransferVarParamsToDevice

subroutine TransferVarParamsToHost

        implicit none

        integer(4) :: istat

   if(ltime) call CpuAdd('start', 'transferVartohost', 1, uout)
        !istat = cudaMemcpy(ro_d,ro,3*np)
        !ro = ro_d
        !virial = virial_d
        !u%tot = utot_d
        u%twob = utwob_d
   if(ltime) call CpuAdd('stop', 'transferVartohost', 1, uout)
end subroutine TransferVarParamsToHost

subroutine TransferDUTotalVarToDevice

        use Molmodule
        implicit none
        integer(4) :: i

   if(ltime) call CpuAdd('start', 'transferDUtoDevice', 1, uout)
   if(ltime) call CpuAdd('start', 'nptm', 2, uout)
        nptm_d = nptm
   if(ltime) call CpuAdd('stop', 'nptm', 2, uout)
   if(ltime) call CpuAdd('start', 'ipnptm', 2, uout)
        ipnptm_d(1:nptm) = ipnptm(1:nptm)
   if(ltime) call CpuAdd('stop', 'ipnptm', 2, uout)
       ! nneighpn_d = nneighpn
   if(ltime) call CpuAdd('start', 'lptm', 2, uout)
        lptm_d = lptm
   if(ltime) call CpuAdd('stop', 'lptm', 2, uout)
   if(ltime) call CpuAdd('start', 'rotm', 2, uout)
      rotm_d(1:3,1:nptm) = rotm(1:3,1:nptm)
       ! rotm_d = rotm
   if(ltime) call CpuAdd('stop', 'rotm', 2, uout)
   !if(ltime) call CpuAdd('start', 'transferPos', 2, uout)
       ! ro_d = ro
   !if(ltime) call CpuAdd('stop', 'transferPos', 2, uout)
   if(ltime) call CpuAdd('start', 'utwobnew', 2, uout)
        utwobnew_d(0:nptpt) = Zero
   if(ltime) call CpuAdd('stop', 'utwobnew', 2, uout)
   if(ltime) call CpuAdd('stop', 'transferDUtoDevice', 1, uout)
        !dutwob_d(0:nptpt) = Zero
        !utwobold_d(0:nptpt) = Zero
        !dutwobold(0:nptpt) = Zero

end subroutine TransferDUTotalVarToDevice

subroutine TransferDUTotalVarToHost

        use Molmodule
        implicit none
        integer(4) :: i
        !logical, intent(inout) :: lhsoverlap
   if(ltime) call CpuAdd('start', 'transferDUToHost', 1, uout)
        dutwobold = utwobold_d
        do i = 1, nptpt
           du%twob(i) = du%twob(i) - dutwobold(i)
        end do
        du%twob(0) = sum(du%twob(1:nptpt))
   if(ltime) call CpuAdd('stop', 'transferDUToHost', 1, uout)


end subroutine TransferDUTotalVarToHost

subroutine TransferStatsToHost

   implicit none
    integer(4) :: istat, ip

    ro = ro_d
    u%tot = utot_d

end subroutine TransferStatsToHost



subroutine GenerateSeeds

   use Random_Module
   implicit none
   integer(4) :: ip
   ix_dev = ix
   iy_dev = iy
   am_dev = am
   iseed_d = iseed
   do ip = 1, np
     seedsnp(ip) = Random_int(iseed)
     seedsnp(ip) = -abs(seedsnp(ip))
   end do
   seeds_d = seedsnp


end subroutine




end module mol_cuda
