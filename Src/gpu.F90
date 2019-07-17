module gpumodule

      implicit none

      real(8),device, allocatable    :: pspart(:)
      real(8), device, allocatable    :: pcharge(:)
      real(8), device, allocatable   :: pmetro(:)
      real(8), device, allocatable   :: ptranx(:)
      real(8), device, allocatable   :: ptrany(:)
      real(8), device, allocatable   :: ptranz(:)
      real(8),device, allocatable    :: dtran(:)
     ! real(8),device, allocatable :: prandom
      !integer(4),parameter    :: isamp = 1   ! sequential 0 or random 1
      integer(4),device              :: iaccept     ! 0 for rejected and 1 for accepteD
      integer(4),device              :: imcaccept = 1
      integer(4),device              :: imcreject = 2
      integer(4),device              :: imcboxreject = 3
      integer(4),device              :: imchsreject = 4
      integer(4),device              :: imchepreject = 5
      integer(4),device              :: imovetype = 1
      integer(4)              :: ispartmove = 1
      integer(4)              :: ichargechangemove = 2
      integer(4), allocatable :: arrevent(:,:,:)
      integer(4), parameter   :: One = 1.0d0
      logical  :: lboxoverlap = .false.! =.true. if box overlap
      logical  :: lhsoverlap  = .false.! =.true. if hard-core overlap
      logical  :: lhepoverlap = .false.! =.true. if hard-external-potential overlap
      real(8)  :: weight      ! nonenergetic weight
    !  real(8),device, allocatable  :: deltaEn(:)
      real(8)                 :: utot
      real(8), device         :: utot_d
      integer(4), device :: blocksize = 512
      integer(4)         :: blocksize_h = 512
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

         use particlemodule, only: np, npt, iptip
         implicit none
         logical, device :: lhsoverlap(np)
         real(8), device :: E_g(np)
         integer(4), device :: ipart
         integer(4) :: i,j

               lhsoverlap = .false.
               E_g = 0.0
               call GenerateRandoms<<<iblock1,512>>>
               call CalcNewPositions<<<iblock1,512>>>
               print *, "New Positions"
               do i = 1, iloops
                  ipart = i
               call CalculateUpperPart<<<iblock2,512>>>(E_g,lhsoverlap,ipart,iloops_d)
               print *, "UpperPart", i
               end do
               print *, "start second loop"
               do j = 1, iloops
                  ipart = j
               call MakeDecision_CalcLowerPart<<<iblock2,512>>>(E_g,lhsoverlap,ipart,iloops_d)
               print *, "LowerPart1", j
               call CalcLowerPart2<<<iblock2,512>>>(E_g, lhsoverlap, ipart, iloops_d)
               print *, "LowerPart2", j
               end do
               !call CountMCsteps(ipmove,iaccept,imovetype)

      end subroutine MCPassAllGPU

      subroutine PrepareMC

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

         if (.not.allocated(dtran)) then
            allocate(dtran(npt))
            dtran = 5.0
         end if

         !if (.not.allocated(lock)) then
         !   allocate(lock(ceiling(real(np)/blocksize_h)))
         !   lock = 0
         !end if
         if (.not.allocated(state)) then
            allocate(state(ceiling(real(np)/blocksize_h)))
            state = .false.
         end if

               iblock1 = np / 512
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

      end subroutine PrepareMC

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

            implicit none
            integer(4) ::  id

            !call curandGenerateUniform(gen,pmetro,numpart)
            !call curandGenerateUniform(gen,ptranx,numpart)
            !call curandGenerateUniform(gen,ptrany,numpart)
            !call curandGenerateUniform(gen,ptranz,numpart)

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
      !!              dtran(npt): maximum displacement for particle type ipt
      !!              iptip(np): particle type of the particles
      attributes(global) subroutine CalcNewPositions

            implicit none
            integer(4)            :: id, id_int
            integer(4),parameter  :: blocksize = 512
            real(8), parameter     :: Half = 0.5d0

            id = (blockidx%x-1)*blocksize + threadIDx%x
            id_int = threadIDx%x

            call syncthreads
            rotm(1,id) = ro(1,id) + (ptranx(id)-Half)*dtran(iptip(id))
            rotm(2,id) = ro(2,id) + (ptrany(id)-Half)*dtran(iptip(id))
            rotm(3,id) = ro(3,id) + (ptranz(id)-Half)*dtran(iptip(id))

            call syncthreads
            call PBC(rotm(1,id),rotm(2,id),rotm(3,id))
            call syncthreads
            !if (id == 1) then
            !   do while(atomiccas(lock,0,1) == 1)
            !   end do
            !end if
            !   call syncthreads
            !if (id_int == 1) then
            !   lock(blockIDx%x) = 0
            !   call threadfence()
            !end if
            !   call syncthreads
      end subroutine CalcNewPositions


      !! subroutine CalculateUpperPart
      !! running on device, calling from device
      !! Calculates the changes in pair energies which are independent on the acceptance of the trial moves
      !! contains:
      !!  subroutines:
      !!  internal parameters:
      !!                      id: global index of thread and index of particle in global list
      !!  global parameters:
      attributes(grid_global) subroutine CalculateUpperPart(E_g,lhsoverlap,ipart,nloop)

         use cooperative_groups
         implicit none
         real(8), shared  ::numblocks
         real(4),shared :: rox(512)
         real(4),shared :: roy(512)
         real(4),shared :: roz(512)
         real(4),shared :: rojx(512)
         real(4),shared :: rojy(512)
         real(4),shared :: rojz(512)
         real(4),shared :: rotmx(512)
         real(4),shared :: rotmy(512)
         real(4),shared :: rotmz(512)
         real(4),shared :: fac(16,16)
         real(4),shared :: rdist(512)
         integer(4),shared :: iptip_s(512)
         integer(4),shared :: iptjp_s(512)
        ! real(4),shared    :: ipcharge_s(1024)
         real(4), shared   :: eps_s(16,16)
         real(4), shared   :: sig_s(16,16)
         real(8), shared   :: E_s(512)
         real(4), shared   :: dx(512)
         real(4), shared   :: dy(512)
         real(4), shared   :: dz(512)
         real(4), shared   :: rsumrad_s(16,16)
         logical, intent(inout)   :: lhsoverlap(512)
         real(8), intent(inout)   :: E_g(*)
         integer(4), shared :: npt_s
         integer(4) :: id, id_int, i, j
         integer(4), intent(in) :: ipart
         type(grid_group) :: gg
         integer(4), shared :: iblock
         integer(4), intent(in) :: nloop
         logical, shared :: lchain_s
         real(8), shared :: bondk_s
         real(8), shared :: bondeq_s
         real(8), shared :: bondp_s

               gg = this_grid()
               id = ((blockIDx%x-1) * blocksize + threadIDx%x)+(np/nloop *(ipart-1))
               id_int = threadIDx%x
               rotmx(id_int) = rotm(1,id)
               rotmy(id_int) = rotm(2,id)
               rotmz(id_int) = rotm(3,id)
               rox(id_int) = ro(1,id)
               roy(id_int) = ro(2,id)
               roz(id_int) = ro(3,id)
               iptip_s(id_int) = iptip(id)
               iptjp_s(id_int) = 0
               E_s(id_int) = E_g(id)
               rdist(id_int) = 0.0
               lhsoverlap(id_int) = .false.
               iptip_s(id_int) = iptip(id)
               lchain_s = lchain_d
              ! ipcharge_s(id_int) = ipcharge(id)
              if (id_int == 1) then
                 numblocks = np / blocksize
               npt_s = npt_d
                  iblock = numblocks / nloop * (ipart - 1)
              end if
              call syncthreads
               if ( id_int <= npt_s) then
                  do i=1, npt_s
                     fac(id_int,i) = facscr_d(id_int,i)
                     sig_s(id_int,i) = sig(id_int,i)
                     eps_s(id_int,i) = eps(id_int,i)
                     rsumrad_s(id_int,i) = rsumrad(id_int,i)
                  end do
               end if



              call syncthreads(gg)

          !! calculate particles that are in other blocks
            do j =blockIDx%x + iblock, (ceiling(numblocks) - 1)
               rojx(id_int) = ro(1,id_int+j*blocksize)
               rojy(id_int) = ro(2,id_int+j*blocksize)
               rojz(id_int) = ro(3,id_int+j*blocksize)
               iptjp_s(id_int) = iptip(id_int+j*blocksize)
               call syncthreads
               do i=1, blocksize
                  !new energy
                     dx(id_int) = rojx(i) - rotmx(id_int)
                     dy(id_int) = rojy(i) - rotmy(id_int)
                     dz(id_int) = rojz(i) - rotmz(id_int)
                  !   rdist(id_int) = (rojx(i)-rotmx(id_int))**2 + (rojy(i)-rotmy(id_int))**2 + (rojz(i) - rotmz(id_int))**2
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptjp_s(i))) then
                        lhsoverlap(id_int) = .true.
                     end if
                     !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptjp_s(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                       ! ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                       ! (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptjp_s(i))* ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                        (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                  !old energy
                     dx(id_int) = rojx(i) - rox(id_int)
                     dy(id_int) = rojy(i) - roy(id_int)
                     dz(id_int) = rojz(i) - roz(id_int)
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptjp_s(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                        ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                        (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
               end do
               call syncthreads
            end do
            !print *, E_s(id_int), id

            !! calculate particles that are in the same block
               call syncthreads
            do i= 1, blocksize
                  !new energy
                  if (id_int < i) then
                     dx(id_int) = rox(i) - rotmx(id_int)
                     dy(id_int) = roy(i) - rotmy(id_int)
                     dz(id_int) = roz(i) - rotmz(id_int)
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptjp_s(i))) then
                        lhsoverlap(id_int) = .true.
                     end if
                     !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptip_s(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                        ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                        (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                  !old energy
                     dx(id_int) = rox(i) - rox(id_int)
                     dy(id_int) = roy(i) - roy(id_int)
                     dz(id_int) = roz(i) - roz(id_int)
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptip_s(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                        ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                        (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                  end if
            end do

               if (lchain_s) then
                     bondk_s = bond_d(1)%k
                     bondeq_s = bond_d(1)%eq
                     bondp_s = bond_d(1)%p
                  do i= 1, 2
                     !ibond_s(i,id_int) = ibond_d(i,id)
                     if (id < ibond_d(i,id)) then
                        dx(id_int) = ro(1, ibond_d(i,id)) - rotmx(id_int)
                        dy(id_int) = ro(2, ibond_d(i,id)) - rotmy(id_int)
                        dz(id_int) = ro(3, ibond_d(i,id)) - rotmz(id_int)
                        call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        rdist(id_int) = sqrt(rdist(id_int))
                       ! print *, "upper: ", ibond_d(i,id), id, rdist(id_int)
                        E_s(id_int) = E_s(id_int) + bondk_s*(rdist(id_int) - bondeq_s)**bondp_s

                        dx(id_int) = ro(1, ibond_d(i,id)) - rox(id_int)
                        dy(id_int) = ro(2, ibond_d(i,id)) - roy(id_int)
                        dz(id_int) = ro(3, ibond_d(i,id)) - roz(id_int)
                        call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        rdist(id_int) = sqrt(rdist(id_int))
                        E_s(id_int) = E_s(id_int) - bondk_s*(rdist(id_int) - bondeq_s)**bondp_s
                     end if

                  end do
               end if

               E_g(id) = E_s(id_int)

      end subroutine CalculateUpperPart





      !attributes(device) subroutine Metropolis(lboxoverlap, lhsoverlap, lhepoverlap, weight, dured)
      attributes(grid_global) subroutine MakeDecision_CalcLowerPart(E_g, lhsoverlap,ipart,nloop)
         use cooperative_groups
         implicit none

         !logical, intent(in)  :: lboxoverlap ! =.true. if box overlap
         !logical, intent(in)  :: lhsoverlap  ! =.true. if hard-core overlap
         !logical, intent(in)  :: lhepoverlap ! =.true. if hard-external-potential overlap
         !real(8), intent(in)  :: weight      ! nonenergetic weight

         !character(40), parameter :: txroutine ='Metropolis'
         real(4) :: fac_metro
         real(8) :: Zero = 0.0d0, One = 1.0d0
         !real(8) :: expmax = 87.0d0
         real(8), shared  ::numblocks
         real(4),shared :: rox(512)
         real(4),shared :: roy(512)
         real(4),shared :: roz(512)
         real(4),shared :: rojx(512)
         real(4),shared :: rojy(512)
         real(4),shared :: rojz(512)
         real(4),shared :: rotmx(512)
         real(4),shared :: rotmy(512)
         real(4),shared :: rotmz(512)
         real(4),shared :: fac(16,16)
         real(4),shared :: rdist(512)
         integer(4),shared :: iptip_s(512)
         integer(4),shared :: iptjp_s(512)
        ! real(4),shared    :: ipcharge_s(1024)
         real(4), shared   :: eps_s(16,16)
         real(4), shared   :: sig_s(16,16)
         real(8), shared   :: E_s(512)
         real(4), shared   :: dx(512)
         real(4), shared   :: dy(512)
         real(4), shared   :: dz(512)
         real(4), shared   :: rsumrad_s(16,16)
         logical, intent(inout)   :: lhsoverlap(512)
         real(8), intent(inout)   :: E_g(*)
         integer(4), intent(in) :: ipart
         integer(4), intent(in) :: nloop
         integer(4), shared :: npt_s
         integer(4) :: id, id_int, i, j
         type(grid_group) :: gg
         logical, shared :: lchain_s
         real(8), shared :: bondk_s
         real(8), shared :: bondeq_s
         real(8), shared :: bondp_s

               gg = this_grid()
               id = ((blockIDx%x-1) * blocksize + threadIDx%x)+(np/nloop*(ipart-1))
               id_int = threadIDx%x
               if (id == 1) print *, "kernel starts"
               call syncthreads(gg)
               rotmx(id_int) = rotm(1,id)
               rotmy(id_int) = rotm(2,id)
               rotmz(id_int) = rotm(3,id)
               rox(id_int) = ro(1,id)
               roy(id_int) = ro(2,id)
               roz(id_int) = ro(3,id)
               iptip_s(id_int) = iptip(id)
               iptjp_s(id_int) = 0
               E_s(id_int) = E_g(id)
               rdist(id_int) = 0.0
               lhsoverlap(id_int) = .false.
               iptip_s(id_int) = iptip(id)
              ! ipcharge_s(id_int) = ipcharge(id)
              if (id_int == 1) then
                 numblocks = np / blocksize
               npt_s = npt_d
              end if
              call syncthreads
               if ( id_int <= npt_s) then
                  do i=1, npt_s
                     fac(id_int,i) = facscr_d(id_int,i)
                     sig_s(id_int,i) = sig(id_int,i)
                     eps_s(id_int,i) = eps(id_int,i)
                     rsumrad_s(id_int,i) = rsumrad(id_int,i)
                  end do
               end if
               lchain_s = lchain_d
               if (lchain_s) then
                     bondk_s = bond_d(1)%k
                     bondeq_s = bond_d(1)%eq
                     bondp_s = bond_d(1)%p
               end if

              call syncthreads(gg)

         do i = 1+(np/nloop*(ipart-1)), np/nloop*ipart  ! es muss np/iloops*ipart sein, 40960 ist auch falsch, es muss np/iloops sein
            if (id == i) then
                  fac_metro = exp(-beta*E_s(id_int))
                  if (fac_metro > One) then
                     iaccept = imcaccept
                  else if (fac_metro > pmetro(id)) then
                     iaccept = imcaccept
                  else
                     iaccept = imcreject
                  end if
                  if (lhsoverlap(id_int) == .true.) then
                      iaccept = imchsreject
                  end if
                  if (iaccept == 1) then
                        ro(1,id) = rotmx(id_int)
                        ro(2,id) = rotmy(id_int)
                        ro(3,id) = rotmz(id_int)
                        utot_d = utot_d + E_s(id_int)
                  end if
                  call countMCsteps(i,iaccept,imovetype)
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
                     dx(id_int) = ro(1,i) - rotmx(id_int)
                     dy(id_int) = ro(2,i) - rotmy(id_int)
                     dz(id_int) = ro(3,i) - rotmz(id_int)
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptip(i))) then
                        lhsoverlap(id_int) = .true.
                     end if
                     !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptip(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptip(i))*&
                     !          ((sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**12 - &
                     !          (sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptip(i))*&
                               ((sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**12 - &
                               (sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**6)
                  !old energy
                     dx(id_int) = ro(1,i) - rox(id_int)
                     dy(id_int) = ro(2,i) - roy(id_int)
                     dz(id_int) = ro(3,i) - roz(id_int)
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptip(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptip(i))*&
                     !          ((sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**12 - &
                     !          (sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptip(i))*&
                               ((sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**12 - &
                               (sig_s(iptip_s(id_int),iptip(i))/rdist(id_int))**6)
            end if

               if (lchain_s) then
                  do j= 1, 2
                     !ibond_s(i,id_int) = ibond_d(i,id)
                     if (i == ibond_d(j,id).AND.i<id) then
                        dx(id_int) = ro(1, i) - rotmx(id_int)
                        dy(id_int) = ro(2, i) - rotmy(id_int)
                        dz(id_int) = ro(3, i) - rotmz(id_int)
                        call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        rdist(id_int) = sqrt(rdist(id_int))
                       ! print *, "lower: ", i, id, rdist(id_int)
                        E_s(id_int) = E_s(id_int) + bondk_s*(rdist(id_int) - bondeq_s)**bondp_s

                        dx(id_int) = ro(1, i) - rox(id_int)
                        dy(id_int) = ro(2, i) - roy(id_int)
                        dz(id_int) = ro(3, i) - roz(id_int)
                        call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        rdist(id_int) = sqrt(rdist(id_int))
                        E_s(id_int) = E_s(id_int) - bondk_s*(rdist(id_int) - bondeq_s)**bondp_s
                     end if

                  end do
               end if
            call syncthreads(gg)
         end do

         !if ( id == 1) print *, "steps", countsteps

      end subroutine MakeDecision_CalcLowerPart


      attributes(grid_global) subroutine CalcLowerPart2(E_g, lhsoverlap, ipart, nloop)

         use cooperative_groups
         implicit none
         real(8), shared  ::numblocks
         real(8), shared  ::numblocks_old
         real(4),shared :: rox(512)
         real(4),shared :: roy(512)
         real(4),shared :: roz(512)
         real(4),shared :: rojx(512)
         real(4),shared :: rojy(512)
         real(4),shared :: rojz(512)
         real(4),shared :: rotmx(512)
         real(4),shared :: rotmy(512)
         real(4),shared :: rotmz(512)
         real(4),shared :: fac(16,16)
         real(4),shared :: rdist(512)
         integer(4),shared :: iptip_s(512)
         integer(4),shared :: iptjp_s(512)
        ! real(4),shared    :: ipcharge_s(1024)
         real(4), shared   :: eps_s(16,16)
         real(4), shared   :: sig_s(16,16)
         real(8), shared   :: E_s(512)
         real(4), shared   :: dx(512)
         real(4), shared   :: dy(512)
         real(4), shared   :: dz(512)
         real(4), shared   :: rsumrad_s(16,16)
         logical, intent(inout)   :: lhsoverlap(512)
         real(8), intent(inout)   :: E_g(*)
         integer(4), shared :: npt_s
         integer(4) :: id, id_int, i, j, q
         integer(4), intent(in) :: ipart
         integer(4), intent(in) :: nloop
         type(grid_group) :: gg
         integer(4) :: ipart_2
         logical, shared :: lchain_s
         real(8), shared :: bondk_s
         real(8), shared :: bondeq_s
         real(8), shared :: bondp_s

         gg = this_grid()
         do ipart_2 = ipart, nloop - 1
               id = ((blockIDx%x-1) * blocksize + threadIDx%x)+(np/nloop*ipart_2)
               id_int = threadIDx%x
               rotmx(id_int) = rotm(1,id)
               rotmy(id_int) = rotm(2,id)
               rotmz(id_int) = rotm(3,id)
               rox(id_int) = ro(1,id)
               roy(id_int) = ro(2,id)
               roz(id_int) = ro(3,id)
               iptip_s(id_int) = iptip(id)
               iptjp_s(id_int) = 0
               E_s(id_int) = E_g(id)
               rdist(id_int) = 0.0
               lhsoverlap(id_int) = .false.
               iptip_s(id_int) = iptip(id)
              ! ipcharge_s(id_int) = ipcharge(id)
              if (id_int == 1) then
                 numblocks = np / blocksize / nloop * ipart
                 numblocks_old = np / blocksize / nloop * (ipart - 1) + 1
               npt_s = npt_d
              end if
              call syncthreads
               if ( id_int <= npt_s) then
                  do i=1, npt_s
                     fac(id_int,i) = facscr_d(id_int,i)
                     sig_s(id_int,i) = sig(id_int,i)
                     eps_s(id_int,i) = eps(id_int,i)
                     rsumrad_s(id_int,i) = rsumrad(id_int,i)
                  end do
               end if
               lchain_s = lchain_d
               if (lchain_s) then
                     bondk_s = bond_d(1)%k
                     bondeq_s = bond_d(1)%eq
                     bondp_s = bond_d(1)%p
               end if



              call syncthreads(gg)

          !! calculate particles that are in other blocks
            do j = numblocks_old, ceiling(numblocks)
               rojx(id_int) = ro(1,id_int+(j-1)*blocksize)
               rojy(id_int) = ro(2,id_int+(j-1)*blocksize)
               rojz(id_int) = ro(3,id_int+(j-1)*blocksize)
               iptjp_s(id_int) = iptip(id_int+(j-1)*blocksize)
               call syncthreads
               do i=1, blocksize
                  !new energy
                     dx(id_int) = rojx(i) - rotmx(id_int)
                     dy(id_int) = rojy(i) - rotmy(id_int)
                     dz(id_int) = rojz(i) - rotmz(id_int)
                  !   rdist(id_int) = (rojx(i)-rotmx(id_int))**2 + (rojy(i)-rotmy(id_int))**2 + (rojz(i) - rotmz(id_int))**2
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptjp_s(i))) then
                        lhsoverlap(id_int) = .true.
                     end if
                     !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptjp_s(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                        ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                        (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                  !old energy
                     dx(id_int) = rojx(i) - rox(id_int)
                     dy(id_int) = rojy(i) - roy(id_int)
                     dz(id_int) = rojz(i) - roz(id_int)
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptjp_s(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptjp_s(i))*&
                        ((sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**12 - &
                        (sig_s(iptip_s(id_int),iptjp_s(i))/rdist(id_int))**6)


               if (lchain_s) then
                  do q= 1, 2
                     !ibond_s(i,id_int) = ibond_d(i,id)
                     if (i+(j-1)*blocksize == ibond_d(q,id)) then
                        dx(id_int) = rojx(id_int) - rotmx(id_int)
                        dy(id_int) = rojy(id_int) - rotmy(id_int)
                        dz(id_int) = rojz(id_int) - rotmz(id_int)
                        call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                        rdist(id_int) = sqrt(rdist(id_int))
                        E_s(id_int) = E_s(id_int) + bondk_s*(rdist(id_int) - bondeq_s)**bondp_s

                        dx(id_int) = rojx(id_int) - rox(id_int)
                        dy(id_int) = rojy(id_int) - roy(id_int)
                        dz(id_int) = rojz(id_int) - roz(id_int)
                        call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
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
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     if (rdist(id_int) < rsumrad_s(iptip_s(id_int),iptjp_s(i))) then
                        lhsoverlap(id_int) = .true.
                     end if
                     !E_s(id_int) = E_s(id_int) + fac(iptip_s(id_int),iptip_s(i))/rdist(id_int) + 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) + 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                        ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                        (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                  !old energy
                     dx(id_int) = rox(i) - rox(id_int)
                     dy(id_int) = roy(i) - roy(id_int)
                     dz(id_int) = roz(i) - roz(id_int)
                     call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                     rdist(id_int) = sqrt(rdist(id_int))
                     !E_s(id_int) = E_s(id_int) - fac(iptip_s(id_int),iptip_s(i))/rdist(id_int) - 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                     !   ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                     !   (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     E_s(id_int) = E_s(id_int) - 4*eps_s(iptip_s(id_int),iptip_s(i))*&
                        ((sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**12 - &
                        (sig_s(iptip_s(id_int),iptip_s(i))/rdist(id_int))**6)
                     if (lchain_s) then
                        do q= 1, 2
                           !ibond_s(i,id_int) = ibond_d(i,id)
                           if (blockIDx%x*blocksize + i == ibond_d(q,id)) then
                              dx(id_int) = rox(i) - rotmx(id_int)
                              dy(id_int) = roy(i) - rotmy(id_int)
                              dz(id_int) = roz(i) - rotmz(id_int)
                              call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
                              rdist(id_int) = sqrt(rdist(id_int))
                              E_s(id_int) = E_s(id_int) + bondk_s*(rdist(id_int) - bondeq_s)**bondp_s

                              dx(id_int) = rox(i) - rox(id_int)
                              dy(id_int) = roy(i) - roy(id_int)
                              dz(id_int) = roz(i) - roz(id_int)
                              call PBCr2Device(dx(id_int),dy(id_int),dz(id_int),rdist(id_int))
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






      subroutine UTotal

         use potentialmodule, only: rcut2, facscr,sig_h,eps_h
         implicit none
            integer(4)  :: ip, jp
            real(4)     :: dr, dx, dy, dz, dr2

            utot = 0.0

            do ip = 1, np - 1
               do jp = ip + 1, np
                  dx = ro_h(1,ip) - ro_h(1,jp)
                  dy = ro_h(2,ip) - ro_h(2,jp)
                  dz = ro_h(3,ip) - ro_h(3,jp)
                  call PBCr2(dx,dy,dz,dr2)
                  if (dr2 > rcut2) cycle
                  dr = sqrt(dr2)
                     !utot = utot + facscr(iptip_h(ip),iptip_h(jp))/dr + 4*eps_h(iptip_h(ip),iptip_h(jp))* &
                     !                      ((sig_h(iptip_h(ip),iptip_h(jp))/dr)**12 - (sig_h(iptip_h(ip),iptip_h(jp))/dr)**6)
                     utot = utot + 4*eps_h(iptip_h(ip),iptip_h(jp))* &
                                           ((sig_h(iptip_h(ip),iptip_h(jp))/dr)**12 - (sig_h(iptip_h(ip),iptip_h(jp))/dr)**6)
               end do
            end do

            if(lchain) then
               do ip = 2, np
                  dx = ro_h(1, ibond(1,ip)) - ro_h(1,ip)
                  dy = ro_h(2, ibond(1,ip)) - ro_h(2,ip)
                  dz = ro_h(3, ibond(1,ip)) - ro_h(3,ip)
                  dr2 = dx**2 + dy**2 + dz**2
                  call PBCr2(dx,dy,dz,dr2)
                  dr = sqrt(dr2)
                  utot = utot + bond(1)%k*((dr - bond(1)%eq)**bond(1)%p)
               end do
            end if
            utot_d = utot


      end subroutine UTotal

      attributes(device) subroutine SyncAcrossBlocks(blockID)

         implicit none
         integer(4) :: blockID

         do while(atomiccas(lock,0,1) == 1)
         end do
         lock = 0
         call threadfence()



      end subroutine SyncAcrossBlocks

end module mcmodule
