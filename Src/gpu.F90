module gpumodule

      use mol_cuda
      use MolModule
      use precision_m
      implicit none

      real(8),device, allocatable    :: pspart_d(:)
      real(8), device, allocatable    :: pcharge_d(:)
      real(fp_kind), device, allocatable   :: pmetro(:)
      real(fp_kind), device, allocatable   :: ptranx(:)
      real(fp_kind), device, allocatable   :: ptrany(:)
      real(fp_kind), device, allocatable   :: ptranz(:)
      real(fp_kind), allocatable   :: pmetro_h(:)
      real(fp_kind), allocatable   :: ptranx_h(:)
      real(fp_kind), allocatable   :: ptrany_h(:)
      real(fp_kind), allocatable   :: ptranz_h(:)
      !real(8),device, allocatable    :: dtran_d(:)
     ! real(8),device, allocatable :: prandom
      !integer(4),parameter    :: isamp = 1   ! sequential 0 or random 1
      integer(4),device,allocatable              :: mcstat_d(:,:)     ! 0 for rejected and 1 for accepteD
      integer(4),device              :: imcaccept_d = 1
      integer(4),device              :: imcreject_d = 2
      integer(4),device              :: imcboxreject_d = 3
      integer(4),device              :: imchsreject_d = 4
      integer(4),device              :: imchepreject_d = 5
      integer(4),device              :: imovetype_d = 1
      !integer(4)              :: ispartmove = 1
      !integer(4)              :: ichargechangemove = 2
      integer(4), allocatable :: arrevent(:,:,:)
      !integer(4), parameter   :: One = 1.0d0
      logical  :: lboxoverlap = .false.! =.true. if box overlap
      !logical  :: lhsoverlap  = .false.! =.true. if hard-core overlap
      logical  :: lhepoverlap = .false.! =.true. if hard-external-potential overlap
      real(8)  :: weight      ! nonenergetic weight
    !  real(8),device, allocatable  :: deltaEn(:)
      !real(8)                 :: utot
      !real(8), device         :: utot_d
      integer(4), device :: blocksize = 256
      integer(4)         :: blocksize_h = 256
      integer(4), device             :: lock = 0
      integer(4), device             :: istat
      integer(4), device             :: goal = 5
      integer(4) :: iloops
      integer(4), device :: iloops_d
      integer(4) :: iblock1, iblock2
      integer(4), device :: iblock2_d

      contains

         !! subroutine MCPass
         !! main routine in mc step
         !! contains
         !!  subroutines:
         !!              CalcNewPositions
         !!              CalculateUpperPart
         !!              MakeDecision_CalcLowerPart
      subroutine MCPassAllGPU

         !use particlemodule, only: np, npt, iptip
         use precision_m
         implicit none
         logical, device :: lhsoverlap(np_d)
         real(fp_kind), device :: E_g(np_d)
         integer(4) :: ipart
         integer(4) :: i,j,istat,ierr,ierra

               lhsoverlap = .false.
               E_g = 0.0
               !call GenerateRandoms_h
               !call GenerateRandoms<<<iblock1,128>>>
               !ierr = cudaGetLastError()
               !ierra = cudaDeviceSynchronize()
               write(*,*) "Randoms"
               !if (ierr /= cudaSuccess) write(*,*) "Sync kernel error: ", cudaGetErrorString(ierr)
               !if (ierra /= cudaSuccess) write(*,*) "Async kernel err: ", cudaGetErrorString(ierra)
               !call CalcNewPositions<<<iblock1,128>>>
               call CalcNewPositions<<<1,1>>>
               ierr = cudaGetLastError()
               ierra = cudaDeviceSynchronize()
               write(*,*) "Positions"
               if (ierr /= cudaSuccess) write(*,*) "Sync kernel error: ", cudaGetErrorString(ierr)
               if (ierra /= cudaSuccess) write(*,*) "Async kernel err: ", cudaGetErrorString(ierra)
               write(*,*) "before upper"
               call CalculateUpperPart<<<iblock1,256>>>(E_g,lhsoverlap)
               ierr = cudaGetLastError()
               ierra = cudaDeviceSynchronize()
               write(*,*) "upper"
               if (ierr /= cudaSuccess) write(*,*) "Sync kernel error: ", cudaGetErrorString(ierr)
               if (ierra /= cudaSuccess) write(*,*) "Async kernel err: ", cudaGetErrorString(ierra)
               write(*,*) "UpperPart"
               write(*,*) "start second loop"
               do j = 1, iloops
                  ipart = j
                  call MakeDecision_CalcLowerPart<<<iblock2,256>>>(E_g,lhsoverlap,ipart,iloops_d)
                  ierr = cudaGetLastError()
                  ierra = cudaDeviceSynchronize()
                  write(*,*) "lower1"
                  if (ierr /= cudaSuccess) write(*,*) "Sync kernel error: ", cudaGetErrorString(ierr)
                  if (ierra /= cudaSuccess) write(*,*) "Async kernel err: ", cudaGetErrorString(ierra)
                  write(*,*) "LowerPart1", j
                  call CalcLowerPart2<<<iblock2,256>>>(E_g, lhsoverlap, ipart, iloops_d)
                  ierr = cudaGetLastError()
                  ierra = cudaDeviceSynchronize()
                  write(*,*) "lower2"
                  if (ierr /= cudaSuccess) write(*,*) "Sync kernel error: ", cudaGetErrorString(ierr)
                  if (ierra /= cudaSuccess) write(*,*) "Async kernel err: ", cudaGetErrorString(ierra)
                  write(*,*) "LowerPart2", j
               end do
               !call CountMCsteps(ipmove,iaccept,imovetype)
               write(*,*) "one step successfull"

      end subroutine MCPassAllGPU

      subroutine PrepareMC_cudaAll

         implicit none

         if (.not.allocated(pmetro)) then
            allocate(pmetro(np))
            pmetro = 0.0
         end if

         if (.not.allocated(ptranx)) then
            allocate(ptranx(np))
            ptranx = 0.0
         end if

         if (.not.allocated(ptrany)) then
            allocate(ptrany(np))
            ptrany = 0.0
         end if

         if (.not.allocated(ptranz)) then
            allocate(ptranz(np))
            ptranz = 0.0
         end if



         if (.not.allocated(pmetro_h)) then
            allocate(pmetro_h(np))
            pmetro_h = 0.0
         end if

         if (.not.allocated(ptranx_h)) then
            allocate(ptranx_h(np))
            ptranx_h = 0.0
         end if

         if (.not.allocated(ptrany_h)) then
            allocate(ptrany_h(np))
            ptrany_h = 0.0
         end if

         if (.not.allocated(ptranz_h)) then
            allocate(ptranz_h(np))
            ptranz_h = 0.0
         end if

         if (.not.allocated(mcstat_d)) then
            allocate(mcstat_d(np,5))
            mcstat_d = 0
         end if
         !if (.not.allocated(dtran_d)) then
         !   allocate(dtran_d(npt))
         !   dtran_d = 0.0
         !end if
         !dtran_d = dtran

         !if (.not.allocated(lock)) then
         !   allocate(lock(ceiling(real(np)/blocksize_h)))
         !   lock = 0
         !end if


         iblock1 = ceiling(real(np) / blocksize_h)
         print *, "iblock1: ", iblock1
         iloops = ceiling(real(np)/20480)
         iloops_d = iloops
         iblock2 = iblock1 / iloops
         iblock2_d = iblock2
         !if (np < blocksize_h) iblock1 = 1
         !      if ( np > 122880) then
                  !iloops = 7
                  !iloops_d = 7
                  !iblock2 = iblock1 / 7
               !else if (np > 102400) then
               !   iloops = 6
               !   iloops_d = 6
               !   iblock2 = iblock1 / 6
               !else if (np > 81920) then
               !   iloops = 5
               !   iloops_d = 5
               !   iblock2 = iblock1 / 5
               !else if (np > 61440) then
               !   iloops = 4
               !   iloops_d = 4
               !   iblock2 = iblock1 / 4
               !else if (np > 40960) then
               !   iloops = 3
               !   iloops_d = 3
               !   iblock2 = iblock1 / 3
               !else if (np > 20480) then
               !   iloops = 2
                  !iloops_d = 2
                  !iblock2 = iblock1 / 2
               !else
               !   iloops = 1
               !   iloops_d = 1
               !   iblock2 = iblock1
               !end if

        ! call GenerateSeeds

      end subroutine PrepareMC_cudaAll

      !! subroutine GenerateRandoms
      !! Generates the random numbers for decision in Metropolis algorithm and for displacements
      !! contains
      !!  subroutines:
      !!              curandGenerateUnifomr
      !!  internal parameters:
      !!                      gen: test parameter
      !!  module parameters:
      !!                      pmetro(np): probability for acceptance of move(i)
      !!                      ptran(np):  probability for displacement of particles
     attributes(global) subroutine GenerateRandoms

            use Random_Module
            implicit none
            integer(4) ::  id


            id = (blockidx%x-1)*blocksize + threadIDx%x
               call Random_d(seeds_d(id),pmetro(id),id)
               call Random_d(seeds_d(id),ptranx(id),id)
               call Random_d(seeds_d(id),ptrany(id),id)
               call Random_d(seeds_d(id),ptranz(id),id)
            !do id = 1, np
            !   pmetro(id) = Random(iseed)
            !   ptranx(id) = Random(iseed)
            !   ptrany(id) = Random(iseed)
            !   ptranz(id) = Random(iseed)
            !end do


      end subroutine GenerateRandoms

      subroutine GenerateRandoms_h

            use Random_Module
            implicit none
            integer(4) ::  id

            do id = 1, np
               ptranx_h(id) = Random_h(iseed)
               ptrany_h(id) = Random_h(iseed)
               ptranz_h(id) = Random_h(iseed)
               pmetro_h(id) = Random_h(iseed)
            end do
               pmetro = pmetro_h
               ptranx = ptranx_h
               ptrany = ptrany_h
               ptranz = ptranz_h


      end subroutine GenerateRandoms_h

      !! subroutine CalcNewPositions
      !! running on device, calling from device
      !! Calculates the new coordinates for all particles
      !! contains:
      !!  subroutines:
      !!              PBC: calculate coordinates for periodic boundary conditions
      !!              syncthreads: synchronizes all threads
      !!  internal parameters:
      !!              id: global index of thread and index of particle in global list
      !!              blocksize: number of threads in one thread block
      !!  global parameters:
      !!              blockidx%x: Index of thread block on the grid
      !!              threadx%x: internal index of thread in its thread block
      !!              ro(1:3,np): coordinates in x,y,z-direction of the particles
      !!              rotm(1:3,np): new coordinates of the particles
      !!              dtran_d(npt): maximum displacement for particle type ipt
      !!              iptpn(np): particle type of the particles
      attributes(global) subroutine CalcNewPositions

            use Random_Module
            use mol_cuda
            implicit none
            integer(4)            :: id, id_int
       !     real(8), parameter     :: Half = 0.5d0

            !id = (blockidx%x-1)*blockDim%x + threadIDx%x

            !rotm_d(1,id) = ro_d(1,id) + (ptranx(id)-Half)*dtran_d(iptpn_d(id))
            !rotm_d(2,id) = ro_d(2,id) + (ptrany(id)-Half)*dtran_d(iptpn_d(id))
            !rotm_d(3,id) = ro_d(3,id) + (ptranz(id)-Half)*dtran_d(iptpn_d(id))
            do id =1, np_d
            rotm_d(1,id) = ro_d(1,id) + (Random_dev(iseed_d)-Half)*dtran_d(iptpn_d(id))
            rotm_d(2,id) = ro_d(2,id) + (Random_dev(iseed_d)-Half)*dtran_d(iptpn_d(id))
            rotm_d(3,id) = ro_d(3,id) + (Random_dev(iseed_d)-Half)*dtran_d(iptpn_d(id))

            call PBC_cuda(rotm_d(1,id),rotm_d(2,id),rotm_d(3,id))
            end do

      end subroutine CalcNewPositions


      !! subroutine CalculateUpperPart
      !! running on device, calling from device
      !! Calculates the changes in pair energies which are independent on the acceptance of the trial moves
      !! contains:
      !!  subroutines:
      !!  internal parameters:
      !!                      id: global index of thread and index of particle in global list
      !!  global parameters:
      attributes(global) subroutine CalculateUpperPart(E_g,lhsoverlap)

         use precision_m
         implicit none
         real(fp_kind)  :: numblocks
         real(fp_kind),shared :: rox(256)
         real(fp_kind),shared :: roy(256)
         real(fp_kind),shared :: roz(256)
         real(fp_kind),shared :: rojx(256)
         real(fp_kind),shared :: rojy(256)
         real(fp_kind),shared :: rojz(256)
         real(fp_kind),shared :: rotmx(256)
         real(fp_kind),shared :: rotmy(256)
         real(fp_kind),shared :: rotmz(256)
         real(fp_kind) :: rdist
         integer(4) :: iptip_s
         integer(4),shared :: iptjp_s(256)
         real(8)   :: E_s
         real(fp_kind)   :: dx
         real(fp_kind)   :: dy
         real(fp_kind)   :: dz
         real(fp_kind), shared   :: rsumrad_s(16,16)
         integer(4) :: ibuf
         logical, intent(inout)   :: lhsoverlap(np_d)
         real(fp_kind), intent(inout)   :: E_g(*)
         integer(4) :: npt_s
         integer(4) :: id, id_int, i, j, jp
         integer(4) :: ictpn_s
         real(fp_kind) :: bondk_s
         real(fp_kind) :: bondeq_s
         real(fp_kind) :: bondp_s
         real(fp_kind) :: clinkk_s
         real(fp_kind) :: clinkeq_s
         real(fp_kind) :: clinkp_s
         real(fp_kind) :: usum

               id = ((blockIDx%x-1) * blocksize + threadIDx%x)
               id_int = threadIDx%x
               if (id <= np_d) then
                  rotmx(id_int) = rotm_d(1,id)
                  rotmy(id_int) = rotm_d(2,id)
                  rotmz(id_int) = rotm_d(3,id)
                  rox(id_int) = ro_d(1,id)
                  roy(id_int) = ro_d(2,id)
                  roz(id_int) = ro_d(3,id)
                  iptip_s = iptpn_d(id)
                  iptjp_s(id_int) = 0
                  ictpn_s = ictpn_d(id)
                  E_s = 0.0
                  rdist = 0.0
                  lhsoverlap(id) = .false.
                    numblocks = np_d / blocksize
                  npt_s = npt_d
                  if ( id_int <= npt_s) then
                     do i=1, npt_s
                        rsumrad_s(id_int,i) = rsumrad(id_int,i)
                     end do
                  end if
                end if


                 call syncthreads

             !! calculate particles that are in other blocks
             do j =blockIDx%x, (ceiling(real(np_d)/blocksize) - 1)
                  call syncthreads
                  if ((id_int + j*blocksize) <= np_d) then
                     rojx(id_int) = ro_d(1,id_int+j*blocksize)
                     rojy(id_int) = ro_d(2,id_int+j*blocksize)
                     rojz(id_int) = ro_d(3,id_int+j*blocksize)
                     iptjp_s(id_int) = iptpn_d(id_int+j*blocksize)
                  end if
                  call syncthreads
                  do i=1, blocksize
                     jp = j*blocksize + i
                     if (jp <= np_d) then
                        !new energy
                        dx = rojx(i) - rotmx(id_int)
                        dy = rojy(i) - rotmy(id_int)
                        dz = rojz(i) - rotmz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        if (rdist < rsumrad_s(iptip_s,iptjp_s(i))) then
                           lhsoverlap(id) = .true.
                        end if
                        call calcUTabplus(id,jp,rdist,usum)
                        E_s = E_s + usum

                     !old energy
                        dx = rojx(i) - rox(id_int)
                        dy = rojy(i) - roy(id_int)
                        dz = rojz(i) - roz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        call calcUTabminus(id,jp,rdist,usum)
                        E_s = E_s - usum
                     end if
                  end do
               end do


              call syncthreads
               !! calculate particles that are in the same block
               do i= 1, blocksize
                     !new energy
                  if (id_int < i) then
                     if ((blockIDx%x-1)*blockDim%x + i <= np_d) then
                        dx = rox(i) - rotmx(id_int)
                        dy = roy(i) - rotmy(id_int)
                        dz = roz(i) - rotmz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        if (rdist < rsumrad_s(iptip_s,iptpn_d(i+(blockIDx%x-1)*blockDim%x))) then
                           lhsoverlap(id) = .true.
                        end if
                        call calcUTabplus(id,i+(blockIDx%x-1)*blockDim%x,rdist,usum)
                        E_s = E_s + usum

                     !old energy
                        dx = rox(i) - rox(id_int)
                        dy = roy(i) - roy(id_int)
                        dz = roz(i) - roz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        call calcUTabminus(id,i+(blockIDx%x-1)*blockDim%x,rdist,usum)
                        E_s = E_s - usum
                     end if
                  end if
               end do

               if (id <= np_d) then
                  if (ictpn_s /= 0) then
                        bondk_s = bond_d_k(ictpn_s)
                        bondeq_s = bond_d_eq(ictpn_s)
                        bondp_s = bond_d_p(ictpn_s)
                     do i= 1, 2
                        if (id < bondnn_d(i,id)) then
                           jp = bondnn_d(i,id)
                           dx = ro_d(1, jp) - rotmx(id_int)
                           dy = ro_d(2, jp) - rotmy(id_int)
                           dz = ro_d(3, jp) - rotmz(id_int)
                           call PBCr2_cuda(dx,dy,dz,rdist)
                           E_s = E_s + bondk_s*(sqrt(rdist) - bondeq_s)**bondp_s

                           dx = ro_d(1, jp) - rox(id_int)
                           dy = ro_d(2, jp) - roy(id_int)
                           dz = ro_d(3, jp) - roz(id_int)
                           call PBCr2_cuda(dx,dy,dz,rdist)
                           E_s = E_s - bondk_s*(sqrt(rdist) - bondeq_s)**bondp_s
                        end if
                     end do
                  end if
                  if (lclink_d) then
                     if (nbondcl_d(id) /= 0) then
                           clinkk_s = clink_d_k
                           clinkeq_s = clink_d_eq
                           clinkp_s = clink_d_p
                        do i=1,nbondcl_d(id)
                           if (id < bondcl_d(i,id)) then
                              jp = bondcl_d(i,id)
                              dx = ro_d(1,jp) - rotmx(id_int)
                              dy = ro_d(2,jp) - rotmy(id_int)
                              dz = ro_d(3,jp) - rotmz(id_int)
                              call PBCr2_cuda(dx,dy,dz,rdist)
                              E_s = E_s + clinkk_s*(sqrt(rdist) - clinkeq_s)**clinkp_s

                              dx = ro_d(1,jp) - rox(id_int)
                              dy = ro_d(2,jp) - roy(id_int)
                              dz = ro_d(3,jp) - roz(id_int)
                              call PBCr2_cuda(dx,dy,dz,rdist)
                              E_s = E_s - clinkk_s*(sqrt(rdist) - clinkeq_s)**clinkp_s
                           end if
                        end do
                     end if
                  end if
               end if


                  E_g(id) = E_s

      end subroutine CalculateUpperPart





      attributes(grid_global) subroutine MakeDecision_CalcLowerPart(E_g, lhsoverlap,ipart,nloop)
         use cooperative_groups
         use precision_m
         use mol_cuda
         use Random_Module
         implicit none

         real(fp_kind) :: fac_metro
         real(8) :: expmax_d = 87.0d0
         real(8)  :: numblocks
         real(fp_kind) :: rox
         real(fp_kind) :: roy
         real(fp_kind) :: roz
         real(fp_kind) :: roix
         real(fp_kind) :: roiy
         real(fp_kind) :: roiz
         real(fp_kind) :: rotmx
         real(fp_kind) :: rotmy
         real(fp_kind) :: rotmz
         real(fp_kind) :: rdistold
         real(fp_kind) :: rdistnew
         real(fp_kind) :: d
         integer(4) :: iptip_s
         real(fp_kind)   :: E_s
         real(fp_kind)   :: dx
         real(fp_kind)   :: dy
         real(fp_kind)   :: dz
         real(fp_kind), shared   :: rsumrad_s(16,16)
         logical, intent(inout)   :: lhsoverlap(np_d)
         real(fp_kind), intent(inout)   :: E_g(*)
         integer(4), value :: ipart
         integer(4), intent(in) :: nloop
         integer(4) :: npt_s
         integer(4) :: id, id_int, i, j , ibuf
         type(grid_group) :: gg
         integer(4) :: ictpn_s
         real(fp_kind) :: bondk_s
         real(fp_kind) :: bondeq_s
         real(fp_kind) :: bondp_s
         real(fp_kind) :: clinkk_s
         real(fp_kind) :: clinkeq_s
         real(fp_kind) :: clinkp_s
         real(8) :: dured
         real(fp_kind) :: usum
         integer(4), ipartmin, ipartmax, idecision
         integer(4), np_s


               gg = this_grid()
               np_s = np_d
               ipartmin = blocksize*iblock2_d*(ipart-1)!np_s/nloop*(ipart-1)
               if (ipart == nloop) then
                  ipartmax = np_s
               else
                  ipartmax = iblock2_d*blocksize*ipart!np_s/nloop*ipart
               end if
               id = ((blockIDx%x-1) * blocksize + threadIDx%x)+ipartmin
               id_int = threadIDx%x
               if (id <= np_s) then
                  rotmx = rotm_d(1,id)
                  rotmy = rotm_d(2,id)
                  rotmz = rotm_d(3,id)
                  rox = ro_d(1,id)
                  roy = ro_d(2,id)
                  roz = ro_d(3,id)
                  iptip_s = iptpn_d(id)
                  E_s = E_g(id)
                    numblocks = np_s / blocksize
                  npt_s = npt_d
                  ictpn_s = ictpn_d(id)
                  if ( id_int <= npt_s) then
                     do i=1, npt_s
                        rsumrad_s(id_int,i) = rsumrad(id_int,i)
                     end do
                  end if
                  if (ictpn_s /= 0) then
                           bondk_s = bond_d_k(ictpn_s)
                           bondeq_s = bond_d_eq(ictpn_s)
                           bondp_s = bond_d_p(ictpn_s)
                  end if
                  if (nbondcl_d(id) /= 0) then
                           clinkk_s = clink_d_k
                           clinkeq_s = clink_d_eq
                           clinkp_s = clink_d_p
                  end if
               end if
               call syncthreads

         do i = 1+ipartmin, ipartmax
            if (id == i) then
                  dured = beta_d*E_s
                  if (lhsoverlap(id) == .true.) then
                      idecision = 4   !imchsreject
                  else
                     if (dured > expmax) then
                        idecision = 2 ! energy rejected
                     else if (dured < -expmax) then
                        idecision = 1   !accepted
                     else
                        fac_metro = exp(-dured)
                        if (fac_metro > One) then
                           idecision = 1 !accepted
                        else if (fac_metro > Random_dev2(iseed2_d)) then
                           idecision = 1 ! accepted
                        else
                           idecision = 2 ! energy rejected
                        end if
                     end if
                     if (idecision == 1) then
                           ro_d(1,id) = rotmx
                           ro_d(2,id) = rotmy
                           ro_d(3,id) = rotmz
                           utot_d = utot_d + E_s
                     end if
                  end if
                  mcstat_d(iptip_s,idecision) = mcstat_d(iptip_s,idecision) + 1
            end if
            call syncthreads(gg)
            roix = ro_d(1,i)
            roiy = ro_d(2,i)
            roiz = ro_d(3,i)
         if ( id <= np_s) then
            if ( id > i) then
                     dx = roix - rotmx
                     dy = roiy - rotmy
                     dz = roiz - rotmz
                     call PBCr2_cuda(dx,dy,dz,rdistnew)
                     if (rdistnew < rsumrad_s(iptip_s,iptpn_d(i))) then
                        lhsoverlap(id) = .true.
                     end if
                     call calcUTabplus(id,i,rdistnew,usum)
                     E_s = E_s + usum
                  !old energy
                     dx = roix - rox
                     dy = roiy - roy
                     dz = roiz - roz
                     call PBCr2_cuda(dx,dy,dz,rdistold)
                     call calcUTabminus(id,i,rdistold,usum)
                     E_s = E_s - usum

               !calculate bonds
               if (ictpn_d(i) /= 0) then
                  do j= 1, 2
                     if (i == bondnn_d(j,id)) then
                        !dx = roix - rotmx
                        !dy = roiy - rotmy
                        !dz = roiz - rotmz
                        !call PBCr2_cuda(dx,dy,dz,rdist)
                        !E_s = E_s + bondk_s*(sqrt(rdist) - bondeq_s)**bondp_s

                        !dx = roix - rox
                        !dy = roiy - roy
                        !dz = roiz - roz
                        !call PBCr2_cuda(dx,dy,dz,rdist)
                        !E_s = E_s - bondk_s*(sqrt(rdist) - bondeq_s)**bondp_s
                        E_s = E_s + bondk_s*((sqrt(rdistnew)-bondeq_s)**bondp_s - &
                           (sqrt(rdistold) - bondeq_s)**bondp_s)
                     end if

                  end do
               end if
               ! calculate crosslinks
               if (lclink_d) then
                  do j=1,nbondcl_d(i)
                     if (id == bondcl_d(j,i)) then
                        !dx = roix - rotmx
                        !dy = roiy - rotmy
                        !dz = roiz - rotmz
                        !call PBCr2_cuda(dx,dy,dz,rdist)
                        !E_s = E_s + clinkk_s*(sqrt(rdist) - clinkeq_s)**clinkp_s

                        !dx = roix - rox
                        !dy = roiy - roy
                        !dz = roiz - roz
                        !call PBCr2_cuda(dx,dy,dz,rdist)
                        !E_s = E_s - clinkk_s*(sqrt(rdist) - clinkeq_s)**clinkp_s
                        E_s = E_s + clinkk_s*((sqrt(rdistnew)-clinkeq_s)**clinkp_s - &
                           (sqrt(rdistold) - clinkeq_s)**clinkp_s)
                     end if
                  end do
               end if
            end if
         end if
            call syncthreads(gg)
      end do


      end subroutine MakeDecision_CalcLowerPart


      attributes(global) subroutine CalcLowerPart2(E_g, lhsoverlap, ipart, nloop)

         !use cooperative_groups
         use precision_m
         implicit none
         real(fp_kind)  ::numblocks
         real(fp_kind)  ::numblocks_old
         real(fp_kind),shared :: rox(256)
         real(fp_kind),shared :: roy(256)
         real(fp_kind),shared :: roz(256)
         real(fp_kind),shared :: rojx(256)
         real(fp_kind),shared :: rojy(256)
         real(fp_kind),shared :: rojz(256)
         real(fp_kind),shared :: rotmx(256)
         real(fp_kind),shared :: rotmy(256)
         real(fp_kind),shared :: rotmz(256)
         !real(fp_kind),shared :: fac(16,16)
         real(fp_kind) :: rdist
         !real(fp_kind) :: d
         integer(4) :: iptip_s
         integer(4) :: iptjp_s(256)
        ! real(fp_kind),shared    :: ipcharge_s(1024)
         !real(fp_kind), shared   :: eps_s(16,16)
         !real(fp_kind), shared   :: sig_s(16,16)
         real(fp_kind)   :: E_s
         real(fp_kind)   :: dx
         real(fp_kind)   :: dy
         real(fp_kind)   :: dz
         real(fp_kind), shared   :: rsumrad_s(16,16)
         logical, intent(inout)   :: lhsoverlap(np_d)
         real(fp_kind), intent(inout)   :: E_g(*)
         integer(4) :: npt_s
         integer(4) :: id, id_int!,
         integer(4) :: i, j, q, jp!, ibuf
         integer(4), value :: ipart
         integer(4), intent(in) :: nloop
         !type(grid_group) :: gg
         integer(4) :: ipart_2, np_s
         integer(4) :: ictpn_s
         real(fp_kind) :: bondk_s
         real(fp_kind) :: bondeq_s
         real(fp_kind) :: bondp_s
         real(fp_kind) :: clinkk_s
         real(fp_kind) :: clinkeq_s
         real(fp_kind) :: clinkp_s
         real(fp_kind) :: usum

         np_s = np_d
         numblocks = iblock2_d * ipart
         numblocks_old = iblock2_d * (ipart - 1) + 1
         npt_s = npt_d

         !gg = this_grid()
         do ipart_2 = ipart, nloop - 1
            id = ((blockIDx%x-1) * blocksize + threadIDx%x)+(iblock2_d*blocksize*ipart_2)
               id_int = threadIDx%x
               if (id <= np_s) then
                  rotmx(id_int) = rotm_d(1,id)
                  rotmy(id_int) = rotm_d(2,id)
                  rotmz(id_int) = rotm_d(3,id)
                  rox(id_int) = ro_d(1,id)
                  roy(id_int) = ro_d(2,id)
                  roz(id_int) = ro_d(3,id)
                  iptip_s = iptpn_d(id)
                  iptjp_s(id_int) = 0
                  E_s = E_g(id)
                  iptip_s = iptpn_d(id)
                  ictpn_s = ictpn_d(id)
                  do j = 1, npt_s
                     do i=1, npt_s
                        rsumrad_s(j,i) = rsumrad(j,i)
                     end do
                  end do
                  if (ictpn_s /= 0) then
                        bondk_s = bond_d_k(ictpn_s)
                        bondeq_s = bond_d_eq(ictpn_s)
                        bondp_s = bond_d_p(ictpn_s)
                  end if
               end if
               call syncthreads




          !! calculate particles that are in other blocks
            do j = numblocks_old, numblocks
               call syncthreads
                  rojx(id_int) = ro_d(1,id_int+(j-1)*blocksize)
                  rojy(id_int) = ro_d(2,id_int+(j-1)*blocksize)
                  rojz(id_int) = ro_d(3,id_int+(j-1)*blocksize)
                  iptjp_s(id_int) = iptpn_d(id_int+(j-1)*blocksize)
               call syncthreads
                  do i=1, blocksize
                     !new energy
                     if (id <= np_s) then
                        dx = rojx(i) - rotmx(id_int)
                        dy = rojy(i) - rotmy(id_int)
                        dz = rojz(i) - rotmz(id_int)
                        !dx = ro_d(i+(j-1)*blocksize) - ro_d(id)
                        !dy = ro_d(i+(j-1)*blocksize) - ro_d(id)
                        !dz = ro_d(i+(j-1)*blocksize) - ro_d(id)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        !if (rdist < rsumrad_s(iptip_s,iptjp_s(i))) then
                        if (rdist < rsumrad_s(iptip_s,iptpn_d(i+(j-1)*blocksize))) then
                           lhsoverlap(id) = .true.
                        end if
                        call calcUTabplus(id,(j-1)*blocksize + i,rdist,usum)
                        E_s = E_s + usum


                     !old energy
                        dx = rojx(i) - rox(id_int)
                        dy = rojy(i) - roy(id_int)
                        dz = rojz(i) - roz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        call calcUTabminus(id,(j-1)*blocksize + i,rdist,usum)
                        E_s = E_s - usum
                     end if
                  end do
            end do
            if (id <= np_s) then
               if (ictpn_s /= 0) then
                  do q= 1, 2
                     if (numblocks_old*blocksize <= bondnn_d(q,id) .and. numblocks*blocksize >= bondnn_d(q,id) ) then
                        dx = rojx(id_int) - rotmx(id_int)
                        dy = rojy(id_int) - rotmy(id_int)
                        dz = rojz(id_int) - rotmz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        rdist = sqrt(rdist)
                        E_s = E_s + bondk_s*(rdist - bondeq_s)**bondp_s

                        dx = rojx(id_int) - rox(id_int)
                        dy = rojy(id_int) - roy(id_int)
                        dz = rojz(id_int) - roz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        rdist = sqrt(rdist)
                        E_s = E_s - bondk_s*(rdist - bondeq_s)**bondp_s
                     end if
                  end do
               end if
               if (lclink_d) then
                  if (nbondcl_d(id) /= 0) then
                        clinkk_s = clink_d_k
                        clinkeq_s = clink_d_eq
                        clinkp_s = clink_d_p
                     do i=1,nbondcl_d(id)
                        if (numblocks_old*blocksize < bondcl_d(i,id).and. numblocks*blocksize >= bondcl_d(i,id)) then
                           jp = bondcl_d(i,id)
                           dx = ro_d(1,jp) - rotmx(id_int)
                           dy = ro_d(2,jp) - rotmy(id_int)
                           dz = ro_d(3,jp) - rotmz(id_int)
                           call PBCr2_cuda(dx,dy,dz,rdist)
                           E_s = E_s + clinkk_s*(sqrt(rdist) - clinkeq_s)**clinkp_s

                           dx = ro_d(1,jp) - rox(id_int)
                           dy = ro_d(2,jp) - roy(id_int)
                           dz = ro_d(3,jp) - roz(id_int)
                           call PBCr2_cuda(dx,dy,dz,rdist)
                           E_s = E_s - clinkk_s*(sqrt(rdist) - clinkeq_s)**clinkp_s
                        end if
                     end do
                  end if
               end if
               E_g(id) = E_s
            end if
               call syncthreads
         end do
      end subroutine CalcLowerPart2

      attributes(device) subroutine calcUTabminus(id,i,rdist,E)

      implicit none
      integer(4), intent(in) :: id
      integer(4), intent(in) :: i
      real(fp_kind), intent(inout) :: E
      real(fp_kind), intent(in) :: rdist
      integer(4) :: ibuf
      real(fp_kind) :: d

        ibuf = iubuflow_d(iptpt_d(iptpn_d(id),iptpn_d(i)))
        !ibuf = 1
         do
            if (rdist >= ubuf_d(ibuf)) exit
            ibuf = ibuf+12
         end do
            d = rdist - ubuf_d(ibuf)
         E = ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
               d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))




      end subroutine calcUTabminus

      attributes(device) subroutine calcUTabplus(id,i,rdist,E)

      implicit none
      integer(4), intent(in) :: id
      integer(4), intent(in) :: i
      real(fp_kind), intent(inout) :: E
      real(fp_kind), intent(in) :: rdist
      integer(4) :: ibuf
      real(fp_kind) :: d

        ibuf = iubuflow_d(iptpt_d(iptpn_d(id),iptpn_d(i)))
        !ibuf = 1
         do
            if (rdist >= ubuf_d(ibuf)) exit
            ibuf = ibuf+12
         end do
            d = rdist - ubuf_d(ibuf)
         E = ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
               d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))




      end subroutine calcUTabplus



end module gpumodule
