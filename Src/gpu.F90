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
      integer(4),device              :: iaccept_d     ! 0 for rejected and 1 for accepteD
      integer(4),device              :: imcaccept_d = 1
      integer(4),device              :: imcreject_d = 2
      integer(4),device              :: imcboxreject_d = 3
      integer(4),device              :: imchsreject_d = 4
      integer(4),device              :: imchepreject_d = 5
      integer(4),device              :: imovetype_d = 1
      integer(4),device              :: inumaccept
      !integer(4)              :: ispartmove = 1
      !integer(4)              :: ichargechangemove = 2
      integer(4), allocatable :: arrevent(:,:,:)
      !integer(4), parameter   :: One = 1.0d0
      logical  :: lboxoverlap = .false.! =.true. if box overlap
      logical  :: lhsoverlap  = .false.! =.true. if hard-core overlap
      logical  :: lhepoverlap = .false.! =.true. if hard-external-potential overlap
      real(8)  :: weight      ! nonenergetic weight
    !  real(8),device, allocatable  :: deltaEn(:)
      !real(8)                 :: utot
      !real(8), device         :: utot_d
      integer(4), device :: blocksize = 128
      integer(4)         :: blocksize_h = 128
      logical, device,allocatable :: state(:)
      integer(4), device             :: lock = 0
      integer(4), device             :: istat
      integer(4), device             :: goal = 5
      integer(4) :: iloops
      integer(4), device :: iloops_d
      integer(4) :: iblock1, iblock2

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
               call CalculateUpperPart<<<iblock2,128>>>(E_g,lhsoverlap)
               ierr = cudaGetLastError()
               ierra = cudaDeviceSynchronize()
               write(*,*) "upper"
               if (ierr /= cudaSuccess) write(*,*) "Sync kernel error: ", cudaGetErrorString(ierr)
               if (ierra /= cudaSuccess) write(*,*) "Async kernel err: ", cudaGetErrorString(ierra)
               write(*,*) "UpperPart", i
               write(*,*) "start second loop"
               do j = 1, iloops
                  ipart = j
               call MakeDecision_CalcLowerPart<<<iblock2,128>>>(E_g,lhsoverlap,ipart,iloops_d)
               ierr = cudaGetLastError()
               ierra = cudaDeviceSynchronize()
               write(*,*) "lower1"
               if (ierr /= cudaSuccess) write(*,*) "Sync kernel error: ", cudaGetErrorString(ierr)
               if (ierra /= cudaSuccess) write(*,*) "Async kernel err: ", cudaGetErrorString(ierra)
               write(*,*) "LowerPart1", j
               call CalcLowerPart2<<<iblock2,128>>>(E_g, lhsoverlap, ipart, iloops_d)
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
         !if (.not.allocated(dtran_d)) then
         !   allocate(dtran_d(npt))
         !   dtran_d = 0.0
         !end if
         !dtran_d = dtran

         !if (.not.allocated(lock)) then
         !   allocate(lock(ceiling(real(np)/blocksize_h)))
         !   lock = 0
         !end if
         if (.not.allocated(state)) then
            allocate(state(ceiling(real(np)/blocksize_h)))
            state = .false.
         end if

         inumaccept = 0

               iblock1 = np / 128
               if (np < 128) iblock1 = 1
               if ( np > 122880) then
                  iloops = 4
                  iloops_d = 4
                  iblock2 = iblock1 / 4
               else if (np > 81920) then
                  iloops = 3
                  iloops_d = 3
                  iblock2 = iblock1 / 3
               else if (np > 40960) then
                  iloops = 2
                  iloops_d = 2
                  iblock2 = iblock1 / 2
               else
                  iloops = 1
                  iloops_d = 1
                  iblock2 = iblock1
               end if

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

         !use cooperative_groups
         use precision_m
         implicit none
         real(fp_kind), shared  ::numblocks
         real(fp_kind),shared :: rox(128)
         real(fp_kind),shared :: roy(128)
         real(fp_kind),shared :: roz(128)
         real(fp_kind),shared :: rojx(128)
         real(fp_kind),shared :: rojy(128)
         real(fp_kind),shared :: rojz(128)
         real(fp_kind),shared :: rotmx(128)
         real(fp_kind),shared :: rotmy(128)
         real(fp_kind),shared :: rotmz(128)
         !real(fp_kind),shared :: fac(16,16)
         real(fp_kind),shared :: rdist(128)
         real(fp_kind) :: d
         integer(4),shared :: iptip_s(128)
         integer(4),shared :: iptjp_s(128)
        ! real(fp_kind),shared    :: ipcharge_s(1024)
         !real(fp_kind), shared   :: eps_s(16,16)
         !real(fp_kind), shared   :: sig_s(16,16)
         real(8), shared   :: E_s(128)
         real(fp_kind), shared   :: dx(128)
         real(fp_kind), shared   :: dy(128)
         real(fp_kind), shared   :: dz(128)
         real(fp_kind), shared   :: rsumrad_s(16,16)
         integer(4) :: ibuf
         logical, intent(inout)   :: lhsoverlap(128)
         real(fp_kind), intent(inout)   :: E_g(*)
         integer(4), shared :: npt_s
         integer(4) :: id, id_int, i, j
         !integer(4), value :: ipart
         !type(grid_group) :: gg
         !integer(4), shared :: iblock
         !integer(4), intent(in) :: nloop
         logical, shared :: lchain_s
         real(fp_kind), shared :: bondk_s
         real(fp_kind), shared :: bondeq_s
         real(fp_kind), shared :: bondp_s

               !gg = this_grid()
               id = ((blockIDx%x-1) * blocksize + threadIDx%x)
               id_int = threadIDx%x
               if (id <= np_d) then
                  rotmx(id_int) = rotm_d(1,id)
                  rotmy(id_int) = rotm_d(2,id)
                  rotmz(id_int) = rotm_d(3,id)
                  rox(id_int) = ro_d(1,id)
                  roy(id_int) = ro_d(2,id)
                  roz(id_int) = ro_d(3,id)
                  iptip_s(id_int) = iptpn_d(id)
                  iptjp_s(id_int) = 0
                  E_s(id_int) = E_g(id)
                  rdist(id_int) = 0.0
                  lhsoverlap(id_int) = .false.
                  iptip_s(id_int) = iptpn_d(id)
                  lchain_s = lchain_d
                 ! ipcharge_s(id_int) = ipcharge(id)
                 if (id_int == 1) then
                    numblocks = np_d / blocksize
                  npt_s = npt_d
                     !iblock = numblocks / nloop * (ipart - 1)
                 end if
                 call syncthreads
                  if ( id_int <= npt_s) then
                     do i=1, npt_s
                        !fac(id_int,i) = facscr_d(id_int,i)
                        !sig_s(id_int,i) = sig(id_int,i)
                        !eps_s(id_int,i) = eps(id_int,i)
                        rsumrad_s(id_int,i) = rsumrad(id_int,i)
                     end do
                  end if



                 call syncthreads

             !! calculate particles that are in other blocks
               do j =blockIDx%x, (ceiling(numblocks) - 1)
                  rojx(id_int) = ro_d(1,id_int+j*blocksize)
                  rojy(id_int) = ro_d(2,id_int+j*blocksize)
                  rojz(id_int) = ro_d(3,id_int+j*blocksize)
                  iptjp_s(id_int) = iptpn_d(id_int+j*blocksize)
                  call syncthreads
                  do i=1, blocksize
                     if (j*blocksize + i <= np_d) then
                     !new energy
                        dx(id_int) = rojx(i) - rotmx(id_int)
                        dy(id_int) = rojy(i) - rotmy(id_int)
                        dz(id_int) = rojz(i) - rotmz(id_int)
                     !   rdist(id_int) = (rojx(i)-rotmx(id_int))**2 + (rojy(i)-rotmy(id_int))**2 + (rojz(i) - rotmz(id_int))**2
                        call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptjp_s(i))) then
                           lhsoverlap(id_int) = .true.
                        end if
                        !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptjp_s(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                          ! ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                          ! (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                        rdist(id_int) = sqrt(rdist(id_int))
                        E_s(id_int) = E_s(id_int) + 4* ((6.0/rdist(id_int))**12 - (6.0/rdist(id_int))**6)
   !                     do
   !                        if (rdist(id_int) >= ubuf_d(ibuf)) exit
   !                        ibuf = ibuf+12
   !                     end do
   !                        d = rdist(id_int) - ubuf_d(ibuf)
   !                     E_s(id_int) = E_s(id_int) + ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
   !                           d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))
                     !old energy
                        dx(id_int) = rojx(i) - rox(id_int)
                        dy(id_int) = rojy(i) - roy(id_int)
                        dz(id_int) = rojz(i) - roz(id_int)
                        call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptjp_s(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                        !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                        !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                        !E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                        !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                        !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                        rdist(id_int) = sqrt(rdist(id_int))
                        E_s(id_int) = E_s(id_int) - 4* ((6.0/rdist(id_int))**12 - (6.0/rdist(id_int))**6)
                        !do
                        !   if (rdist(id_int) >= ubuf_d(ibuf)) exit
                        !   ibuf = ibuf+12
                        !end do
                        !   d = rdist(id_int) - ubuf_d(ibuf)
                        !E_s(id_int) = E_s(id_int) - (ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                        !      d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6))))))
                     end if
                  end do
                  call syncthreads
               end do
               !print *, E_s(id_int), id

               !! calculate particles that are in the same block
                  call syncthreads
               do i= 1, blocksize
                     !new energy
                     if (id_int < i) then
                        if ((blockIDx%x-1)*blockDim%x + i <= np_d) then
                        dx(id_int) = rox(i) - rotmx(id_int)
                        dy(id_int) = roy(i) - rotmy(id_int)
                        dz(id_int) = roz(i) - rotmz(id_int)
                        call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptjp_s(i))) then
                           lhsoverlap(id_int) = .true.
                        end if
                        !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptip_s(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                        !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                        !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                        !E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                        !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                        !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                        rdist(id_int) = sqrt(rdist(id_int))
                        E_s(id_int) = E_s(id_int) + 4* ((6.0/rdist(id_int))**12 - (6.0/rdist(id_int))**6)
                        !do
                        !   if (rdist(id_int) >= ubuf_d(ibuf)) exit
                        !   ibuf = ibuf+12
                        !end do
                        !   d = rdist(id_int) - ubuf_d(ibuf)
                        !E_s(id_int) = E_s(id_int) + ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                        !      d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))
                     !old energy
                        dx(id_int) = rox(i) - rox(id_int)
                        dy(id_int) = roy(i) - roy(id_int)
                        dz(id_int) = roz(i) - roz(id_int)
                        call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptip_s(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                        !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                        !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                        !E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                        !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                        !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                        rdist(id_int) = sqrt(rdist(id_int))
                        print *, "tm: ", id, rotmx(id_int), rotmy(id_int), rotmz(id_int)
                        print *, "ol ", i, ro_d(1,i), ro_d(2,i), ro_d(3,i)
                        print *, id, i, rdist(id_int)
                        E_s(id_int) = E_s(id_int) - 4* ((6.0/rdist(id_int))**12 - (6.0/rdist(id_int))**6)
                        !do
                        !   if (rdist(id_int) >= ubuf_d(ibuf)) exit
                        !   ibuf = ibuf+12
                        !end do
                        !   d = rdist(id_int) - ubuf_d(ibuf)
                        !E_s(id_int) = E_s(id_int) - (ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                        !      d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6))))))
                     end if
                  end if
               end do

                  if (lchain_s) then
                        bondk_s = bond_d_k(1)
                        bondeq_s = bond_d_eq(1)
                        bondp_s = bond_d_p(1)
                     do i= 1, 2
                        !ibond_s(i,id_int) = ibond_d(i,id)
                        if (id < bondnn_d(i,id)) then
                           dx(id_int) = ro_d(1, bondnn_d(i,id)) - rotmx(id_int)
                           dy(id_int) = ro_d(2, bondnn_d(i,id)) - rotmy(id_int)
                           dz(id_int) = ro_d(3, bondnn_d(i,id)) - rotmz(id_int)
                           call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                           rdist(id_int) = sqrt(rdist(id_int))
                          ! print *, "upper: ", bond_d(i,id), id, rdist(id_int)
                           E_s(id_int) = E_s(id_int) + bondk_s*(rdist(id_int) - bondeq_s)**bondp_s

                           dx(id_int) = ro_d(1, bondnn_d(i,id)) - rox(id_int)
                           dy(id_int) = ro_d(2, bondnn_d(i,id)) - roy(id_int)
                           dz(id_int) = ro_d(3, bondnn_d(i,id)) - roz(id_int)
                           call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                           rdist(id_int) = sqrt(rdist(id_int))
                           E_s(id_int) = E_s(id_int) - bondk_s*(rdist(id_int) - bondeq_s)**bondp_s
                        end if

                     end do
                  end if

                  E_g(id) = E_s(id_int)
            end if

      end subroutine CalculateUpperPart





      !attributes(device) subroutine Metropolis(lboxoverlap, lhsoverlap, lhepoverlap, weight, dured)
      attributes(grid_global) subroutine MakeDecision_CalcLowerPart(E_g, lhsoverlap,ipart,nloop)
         use cooperative_groups
         use precision_m
         use mol_cuda
         use Random_Module
         implicit none

         !logical, intent(in)  :: lboxoverlap ! =.true. if box overlap
         !logical, intent(in)  :: lhsoverlap  ! =.true. if hard-core overlap
         !logical, intent(in)  :: lhepoverlap ! =.true. if hard-external-potential overlap
         !real(8), intent(in)  :: weight      ! nonenergetic weight

         !character(40), parameter :: txroutine ='Metropolis'
         real(fp_kind) :: fac_metro
         !real(8) :: Zero = 0.0d0, One = 1.0d0
         real(8) :: expmax_d = 87.0d0
         real(8)  ::numblocks
         real(fp_kind),shared :: rox(128)
         real(fp_kind),shared :: roy(128)
         real(fp_kind),shared :: roz(128)
         real(fp_kind),shared :: rojx(128)
         real(fp_kind),shared :: rojy(128)
         real(fp_kind),shared :: rojz(128)
         real(fp_kind),shared :: rotmx(128)
         real(fp_kind),shared :: rotmy(128)
         real(fp_kind),shared :: rotmz(128)
         !real(fp_kind),shared :: fac(16,16)
         real(fp_kind) :: rdist
         real(fp_kind) :: d
         integer(4) :: iptip_s
        ! real(fp_kind),shared    :: ipcharge_s(1024)
         !real(fp_kind), shared   :: eps_s(16,16)
         !real(fp_kind), shared   :: sig_s(16,16)
         real(fp_kind), shared   :: E_s(128)
         real(fp_kind)   :: dx
         real(fp_kind)   :: dy
         real(fp_kind)   :: dz
         real(fp_kind), shared   :: rsumrad_s(16,16)
         logical, intent(inout)   :: lhsoverlap(128)
         real(fp_kind), intent(inout)   :: E_g(*)
         integer(4), value :: ipart
         integer(4), intent(in) :: nloop
         integer(4) :: npt_s
         integer(4) :: id, id_int, i, j , ibuf
         type(grid_group) :: gg
         logical:: lchain_s
         real(fp_kind) :: bondk_s
         real(fp_kind) :: bondeq_s
         real(fp_kind) :: bondp_s
         real(8) :: ubuf_aux
         real(8) :: dured


               gg = this_grid()
               id = ((blockIDx%x-1) * blocksize + threadIDx%x)+(np_d/nloop*(ipart-1))
               id_int = threadIDx%x
               if (id <= np_d) then
               if (id == 1) print *, "kernel starts"
               call syncthreads(gg)
               rotmx(id_int) = rotm_d(1,id)
               rotmy(id_int) = rotm_d(2,id)
               rotmz(id_int) = rotm_d(3,id)
               rox(id_int) = ro_d(1,id)
               roy(id_int) = ro_d(2,id)
               roz(id_int) = ro_d(3,id)
               iptip_s = iptpn_d(id)
               E_s(id_int) = E_g(id)
               rdist = 0.0
               lhsoverlap(id_int) = .false.
              ! ipcharge_s(id_int) = ipcharge(id)
              if (id_int == 1) then
                 numblocks = np_d / blocksize
               npt_s = npt_d
              end if
              call syncthreads
               if ( id_int <= npt_s) then
                  do i=1, npt_s
                     !fac(id_int,i) = facscr_d(id_int,i)
                     !sig_s(id_int,i) = sig(id_int,i)
                     !eps_s(id_int,i) = eps(id_int,i)
                     rsumrad_s(id_int,i) = rsumrad(id_int,i)
                  end do
               end if
               lchain_s = lchain_d
               if (lchain_s) then
                     bondk_s = bond_d_k(1)
                     bondeq_s = bond_d_eq(1)
                     bondp_s = bond_d_p(1)
               end if

              call syncthreads(gg)

         do i = 1+(np_d/nloop*(ipart-1)), np_d/nloop*ipart  ! es muss np/iloops*ipart sein, 40960 ist auch falsch, es muss np/iloops sein
            if (id == i) then
                  print *, "Energie: ", id, E_s(id_int)
                  dured = beta_d*E_s(id_int)
                  if (lhsoverlap(id_int) == .true.) then
                      iaccept_d = imchsreject_d
                  !else if (fac_metro > pmetro(id)) then
                  else if (dured > expmax) then
                     iaccept_d = imcreject_d
                  else if (dured < -expmax) then
                     iaccept_d = imcaccept_d
                  else
                     fac_metro = exp(-dured)
                     if (fac_metro > One) then
                        iaccept_d = imcaccept_d
                     else if (fac_metro > Random_dev2(iseed2_d)) then
                        iaccept_d = imcaccept_d
                     else
                        iaccept_d = imcreject_d
                     end if
                  end if
                  !if (lhsoverlap(id_int) == .true.) then
                  !    iaccept_d = imchsreject_d
                  !end if
                  if (iaccept_d == 1) then
                        ro_d(1,id) = rotmx(id_int)
                        ro_d(2,id) = rotmy(id_int)
                        ro_d(3,id) = rotmz(id_int)
                        utot_d = utot_d + E_s(id_int)
                        inumaccept = inumaccept + 1
                  end if
                  print *, "inumaccept: ", inumaccept
                  print *, "utot_d: ", utot_d
                  !call countMCsteps(i,iaccept_d,imovetype_d)
                  !countsteps = countsteps + 1
            end if
            call syncthreads(gg)
            !if(id_int == 1) then
               !state(blockIDx%x) = .true.
            !   istat = atomicAdd(lock,1)
            !   print *, "lock", lock, id
            !   do while(atomiccas(lock,goal,goal)/= goal)
               !istat = atomiccas(lock,goal,goal)
               !print *, "lock", lock, id
            !   end do
            !   goal = goal + 5
                  !state(blockIDx%x) = 1
                  !print *, "lock", lock
                  !call threadfence()
                  !lock = 0
            !end if
            !call syncthreads
            if ( id > i) then
                     dx = ro_d(1,i) - rotmx(id_int)
                     dy = ro_d(2,i) - rotmy(id_int)
                     dz = ro_d(3,i) - rotmz(id_int)
                     call PBCr2_cuda(dx,dy,dz,rdist)
                     if (rdist < rsumrad_s(iptip_s,iptpn_d(i))) then
                        lhsoverlap(id_int) = .true.
                     end if
                     !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptip(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptip(i))*&
                     !          ((sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**12 - &
                     !          (sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**6)
                     !E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptpn_d(i))*&
                     !          ((sig_s(iptip_s(id_int),iptpn_d(i))/rdist(id_int))**12 - &
                     !          (sig_s(iptip_s(id_int),iptpn_d(i))/rdist(id_int))**6)
                     rdist = sqrt(rdist)
                     print *, id, rotmx(id_int), rotmy(id_int), rotmz(id_int)
                     print *, i, ro_d(1,i), ro_d(2,i), ro_d(3,i)
                     print *, id, i, rdist
                     E_s(id_int) = E_s(id_int) + 4* ((6.0/rdist)**12 - (6.0/rdist)**6)
                     !else if (rdist > rcut2_d) then
                        !nothing
                     !else
                     !   ibuf = iubuflow_d(iptpt_d(iptip_s,iptpn_d(i)))
                        !ibuf = 1
                        !print *, ibuf
                     !   do
                     !      if (rdist >= ubuf_d(ibuf)) exit
                     !      ibuf = ibuf+12
                     !   end do
                     !      d = rdist - ubuf_d(ibuf)
                     !   E_s(id_int) = E_s(id_int) + ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                     !         d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))
                     !end if
                  !old energy
                     dx = ro_d(1,i) - rox(id_int)
                     dy = ro_d(2,i) - roy(id_int)
                     dz = ro_d(3,i) - roz(id_int)
                     call PBCr2_cuda(dx,dy,dz,rdist)
                     !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptip(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptip(i))*&
                     !          ((sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**12 - &
                     !          (sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**6)
                     !E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptpn_d(i))*&
                     !          ((sig_s(iptip_s(id_int),iptpn_d(i))/rdist(id_int))**12 - &
                     !          (sig_s(iptip_s(id_int),iptpn_d(i))/rdist(id_int))**6)
                     rdist = sqrt(rdist)
                     E_s(id_int) = E_s(id_int) - 4* ((6.0/rdist)**12 - (6.0/rdist)**6)
                     !print *, id, i, E_s(id_int)
                    ! if ( rdist > rcut2_d) then
                    !    ibuf = iubuflow_d(iptpt_d(iptip_s,iptpn_d(i)))
                    ! else
                    !    do
                    !       if (rdist >= ubuf_d(ibuf)) exit
                     !      ibuf = ibuf+12
                     !   end do
                     !      d = rdist - ubuf_d(ibuf)
                     !   E_s(id_int) = E_s(id_int) - (ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                     !         d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6))))))
                     !end if
            end if

               if (lchain_s) then
                  do j= 1, 2
                     !ibond_s(i,id_int) = ibond_d(i,id)
                     if (i == bondnn_d(j,id).AND.i<id) then
                        dx = ro_d(1, i) - rotmx(id_int)
                        dy = ro_d(2, i) - rotmy(id_int)
                        dz = ro_d(3, i) - rotmz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        rdist = sqrt(rdist)
                       ! print *, "lower: ", i, id, rdist(id_int)
                        E_s(id_int) = E_s(id_int) + bondk_s*(rdist - bondeq_s)**bondp_s

                        dx = ro_d(1, i) - rox(id_int)
                        dy = ro_d(2, i) - roy(id_int)
                        dz = ro_d(3, i) - roz(id_int)
                        call PBCr2_cuda(dx,dy,dz,rdist)
                        rdist = sqrt(rdist)
                        E_s(id_int) = E_s(id_int) - bondk_s*(rdist - bondeq_s)**bondp_s
                     end if

                  end do
               end if
            call syncthreads(gg)
         end do
      end if

         !if ( id == 1) print *, "steps", countsteps

      end subroutine MakeDecision_CalcLowerPart


      attributes(global) subroutine CalcLowerPart2(E_g, lhsoverlap, ipart, nloop)

         !use cooperative_groups
         use precision_m
         implicit none
         real(fp_kind), shared  ::numblocks
         real(fp_kind), shared  ::numblocks_old
         real(fp_kind),shared :: rox(128)
         real(fp_kind),shared :: roy(128)
         real(fp_kind),shared :: roz(128)
         real(fp_kind),shared :: rojx(128)
         real(fp_kind),shared :: rojy(128)
         real(fp_kind),shared :: rojz(128)
         real(fp_kind),shared :: rotmx(128)
         real(fp_kind),shared :: rotmy(128)
         real(fp_kind),shared :: rotmz(128)
         !real(fp_kind),shared :: fac(16,16)
         real(fp_kind), shared :: rdist(128)
         !real(fp_kind) :: d
         integer(4),shared :: iptip_s(128)
         integer(4),shared :: iptjp_s(128)
        ! real(fp_kind),shared    :: ipcharge_s(1024)
         !real(fp_kind), shared   :: eps_s(16,16)
         !real(fp_kind), shared   :: sig_s(16,16)
         real(8), shared   :: E_s(128)
         real(fp_kind), shared   :: dx(128)
         real(fp_kind), shared   :: dy(128)
         real(fp_kind), shared   :: dz(128)
         real(fp_kind), shared   :: rsumrad_s(16,16)
         logical, intent(inout)   :: lhsoverlap(128)
         real(fp_kind), intent(inout)   :: E_g(*)
         integer(4), shared :: npt_s
         integer(4) :: id, id_int!,
         integer(4) :: i, j, q!, ibuf
         integer(4), value :: ipart
         integer(4), intent(in) :: nloop
         !type(grid_group) :: gg
         integer(4) :: ipart_2
         logical, shared :: lchain_s
         real(fp_kind), shared :: bondk_s
         real(fp_kind), shared :: bondeq_s
         real(fp_kind), shared :: bondp_s


         !gg = this_grid()
         do ipart_2 = ipart, nloop - 1
               id = ((blockIDx%x-1) * blocksize + threadIDx%x)+(np_d/nloop*ipart_2)
               id_int = threadIDx%x
               rotmx(id_int) = rotm_d(1,id)
               rotmy(id_int) = rotm_d(2,id)
               rotmz(id_int) = rotm_d(3,id)
               rox(id_int) = ro_d(1,id)
               roy(id_int) = ro_d(2,id)
               roz(id_int) = ro_d(3,id)
               iptip_s(id_int) = iptpn_d(id)
               iptjp_s(id_int) = 0
               E_s(id_int) = E_g(id)
               rdist(id_int) = 0.0
               lhsoverlap(id_int) = .false.
               iptip_s(id_int) = iptpn_d(id)
              ! ipcharge_s(id_int) = ipcharge(id)
              if (id_int == 1) then
                 numblocks = np_d / blocksize / nloop * ipart
                 numblocks_old = np_d / blocksize / nloop * (ipart - 1) + 1
               npt_s = npt_d
              end if
              call syncthreads
               if ( id_int <= npt_s) then
                  do i=1, npt_s
                     !fac(id_int,i) = facscr_d(id_int,i)
                     !sig_s(id_int,i) = sig(id_int,i)
                     !eps_s(id_int,i) = eps(id_int,i)
                     rsumrad_s(id_int,i) = rsumrad(id_int,i)
                  end do
               end if
               lchain_s = lchain_d
               if (lchain_s) then
                     bondk_s = bond_d_k(1)
                     bondeq_s = bond_d_eq(1)
                     bondp_s = bond_d_p(1)
               end if



         !     call syncthreads(gg)

          !! calculate particles that are in other blocks
            do j = numblocks_old, ceiling(numblocks)
               rojx(id_int) = ro_d(1,id_int+(j-1)*blocksize)
               rojy(id_int) = ro_d(2,id_int+(j-1)*blocksize)
               rojz(id_int) = ro_d(3,id_int+(j-1)*blocksize)
               iptjp_s(id_int) = iptpn_d(id_int+(j-1)*blocksize)
               call syncthreads
               do i=1, blocksize
                  !new energy
                     dx(id_int) = rojx(i) - rotmx(id_int)
                     dy(id_int) = rojy(i) - rotmy(id_int)
                     dz(id_int) = rojz(i) - rotmz(id_int)
                  !   rdist(id_int) = (rojx(i)-rotmx(id_int))**2 + (rojy(i)-rotmy(id_int))**2 + (rojz(i) - rotmz(id_int))**2
                     call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptjp_s(i))) then
                        lhsoverlap(id_int) = .true.
                     end if
                     !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptjp_s(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                     !rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                     !do
                     !   if (rdist(id_int) >= ubuf_d(ibuf)) exit
                     !   ibuf = ibuf+12
                     !end do
                     !   d = rdist(id_int) - ubuf_d(ibuf)
                     !E_s(id_int) = E_s(id_int) + ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                     !      d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))
                  !old energy
                     dx(id_int) = rojx(i) - rox(id_int)
                     dy(id_int) = rojy(i) - roy(id_int)
                     dz(id_int) = rojz(i) - roz(id_int)
                     call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptjp_s(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                     !rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                     !do
                     !   if (rdist(id_int) >= ubuf_d(ibuf)) exit
                     !   ibuf = ibuf+12
                     !end do
                     !   d = rdist(id_int) - ubuf_d(ibuf)
                     !E_s(id_int) = E_s(id_int) + ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                     !      d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))


               if (lchain_s) then
                  do q= 1, 2
                     !ibond_s(i,id_int) = ibond_d(i,id)
                     if (i+(j-1)*blocksize == bondnn_d(q,id)) then
                        dx(id_int) = rojx(id_int) - rotmx(id_int)
                        dy(id_int) = rojy(id_int) - rotmy(id_int)
                        dz(id_int) = rojz(id_int) - rotmz(id_int)
                        call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        rdist(id_int) = sqrt(rdist(id_int))
                        E_s(id_int) = E_s(id_int) + bondk_s*(rdist(id_int) - bondeq_s)**bondp_s

                        dx(id_int) = rojx(id_int) - rox(id_int)
                        dy(id_int) = rojy(id_int) - roy(id_int)
                        dz(id_int) = rojz(id_int) - roz(id_int)
                        call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        rdist(id_int) = sqrt(rdist(id_int))
                        E_s(id_int) = E_s(id_int) - bondk_s*(rdist(id_int) - bondeq_s)**bondp_s
                     end if
                  end do
               end if
           end do
               call syncthreads
            end do
               call syncthreads
            do i= 1, blocksize
                  !new energy
                  if (id_int > i) then
                     dx(id_int) = rox(i) - rotmx(id_int)
                     dy(id_int) = roy(i) - rotmy(id_int)
                     dz(id_int) = roz(i) - rotmz(id_int)
                     call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptjp_s(i))) then
                        lhsoverlap(id_int) = .true.
                     end if
                     !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptip_s(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     !rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     !do
                     !   if (rdist(id_int) >= ubuf_d(ibuf)) exit
                     !   ibuf = ibuf+12
                     !end do
                     !   d = rdist(id_int) - ubuf_d(ibuf)
                     !E_s(id_int) = E_s(id_int) + ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                     !      d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))
                  !old energy
                     dx(id_int) = rox(i) - rox(id_int)
                     dy(id_int) = roy(i) - roy(id_int)
                     dz(id_int) = roz(i) - roz(id_int)
                     call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptip_s(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     !rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     !do
                     !   if (rdist(id_int) >= ubuf_d(ibuf)) exit
                     !   ibuf = ibuf+12
                     !end do
                     !   d = rdist(id_int) - ubuf_d(ibuf)
                     !E_s(id_int) = E_s(id_int) + ubuf_d(ibuf+1)+d*(ubuf_d(ibuf+2)+d*(ubuf_d(ibuf+3)+ &
                     !      d*(ubuf_d(ibuf+4)+d*(ubuf_d(ibuf+5)+d*ubuf_d(ibuf+6)))))
                     if (lchain_s) then
                        do q= 1, 2
                           !ibond_s(i,id_int) = ibond_d(i,id)
                           if (blockIDx%x*blocksize + i == bondnn_d(q,id)) then
                              dx(id_int) = rox(i) - rotmx(id_int)
                              dy(id_int) = roy(i) - rotmy(id_int)
                              dz(id_int) = roz(i) - rotmz(id_int)
                              call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                              rdist(id_int) = sqrt(rdist(id_int))
                              E_s(id_int) = E_s(id_int) + bondk_s*(rdist(id_int) - bondeq_s)**bondp_s

                              dx(id_int) = rox(i) - rox(id_int)
                              dy(id_int) = roy(i) - roy(id_int)
                              dz(id_int) = roz(i) - roz(id_int)
                              call PBCr2_cuda(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                              rdist(id_int) = sqrt(rdist(id_int))
                              E_s(id_int) = E_s(id_int) - bondk_s*(rdist(id_int) - bondeq_s)**bondp_s
                           end if
                        end do
                     end if
                  end if
            end do

               E_g(id) = E_s(id_int)
         end do
      end subroutine CalcLowerPart2

end module gpumodule
