      Program model
      Implicit Real*8 (a-h, o-z)
      parameter ( idim=171, idm=3*idim, iln=37064, ilen=1 )
      parameter (idg2=64*32,idg4=90*46)
      integer nrc,nrc1
      real*8 psi(idm+1),pso(idm),f(idm),omm(idm),omp(idm),buf(idm),z1
      real*8 om2(idim),ps(idim),om(idim),tauR,tauE,tauH,tH0,ar1,ar2,&
     & alphaR,alphaE,miuH,evl(idim),top(idim),tem(idm),va(iln), &
     & tau,tau1,tau2,rpar,tpar
      real*4 sp2gr2(idg2,idim),sp2gr4(idg4,idim),psig2(idg2/2)
      real*4 psig4(idg4/2)
      real*8 blfr(idim,idim)
      real*8 pinv(idim,6),e0,e1,e2,mi,di
      real*4 psiout(idm,500),s3,enerk
      integer*2 sc(iln),fc(iln),j1,j2,j3,j4,l1,l2,l3,l4,a,b
      real*8, allocatable :: ourtop(:)

      pi=4.0d0*dasin(1.0d0)

      open(2,file='model.prm')
      read (2,*) time
      read (2,*) tau
      read (2,*) out
      read (2,*) tauR
      read (2,*) tauE
      read (2,*) tauH
      read (2,*) tH0
      read (2,*) ar1
      read (2,*) ar2
      read (2,*) rpar
      read (2,*) tpar
      read (2,*) zcr
      read (2,*) iout1
      read (2,*) iout2
      read (2,*) iout3
      close(2)

!     Laplace eigen values
      call eform(evl)


!     arrayas for spectral to grid transformations

      if (iout3.eq.1) then
        open (22,file='sp2gr44.std',form='UNFORMATTED',&
      & access='DIRECT',recl=idg4 )
        do ii=1,idim
        read (Unit = 22,rec=ii ) (sp2gr4(i,ii),i=1,idg4)
        end do
        close(22)
      end if 

      if (iout2.eq.1) then
        open (22,file='sp2gr22.std',form='UNFORMATTED',&
      & access='DIRECT',recl=idg2 )
        do ii=1,idim
        read (Unit = 22,rec=ii ) (sp2gr2(i,ii),i=1,idg2)
        end do
        close(22)
      end if


!     dissipation 

      alphaR=1.0d0/(tauR*pi)
      alphaE=1.0d0/(tauE*pi)
      miuH=1.0d0/(21.0d0**4*22.0d0**4*tauH*pi)

      time=time*pi
      nrc=1
      nrc1=nrc
      nrc2=1
      nrc3=1

      tau=tau*pi
      nout=out*(pi/tau)+0.5
      nout1=nout/1.

      nstep=int(time/tau+0.5)
      if(nout.lt.1) nout=1


!     initial condition from text
      open (21,file='init')
      do i=1,idm
         read(21,*) psi(i)
      end do
      read(21,*) gotime
      gotime=0.
      close(21)

!     initial condition  from real*8 direct access data, (UPO # iii)
!     period=int(psi(idm+1))
!     tau=tau*psi(idm+1)/period

!      open (21,file='perp.dt',form='UNFORMATTED',&
!      & access='DIRECT',recl=(idm+1)*2
!      read (Unit = 21,rec=iii ) (psi(i),i=1,idm+1)
!      close(21)


!     forcing
      open(22,file='f0avr.dat')
      do i=1,idm
         read(22,*) tem(i)
      end do
      close(22)

!     orography
      open (21,file='topmm.dat')
      do i=1,idim
         read(21,*) top(i)
         top(i)=tpar*top(i)/tH0
      end do
      close(21)
! 
      allocate(ourtop(idg4/2))
      ourtop(:)=9.d0
      do j=1,idim
         do k0=1,idg4/2
            ourtop(k0)=ourtop(k0)+sp2gr4(k0+idg4/2,j)*top(j)
         end do
      end do
      do k0=1,idg4/2
         write(121,*) ourtop(k0)
      end do

      istep0=1

      write(*,*) istep0,nstep

!     numbers of harmonics in triad interactions
      open (21,file='cc-18b.std',form='UNFORMATTED', &
      & access='DIRECT',recl=ilen )
      do i=1,iln
         read (Unit = 21,rec=i ) fc(i),sc(i)
      end do
      close(21)

!     values of coefficients in triad intercations
      open (21,file='va-18b.std',form='UNFORMATTED',&
      & access='DIRECT',recl=2*ilen*iln )
      read (Unit = 21,rec=1 ) (va(i),i=1,iln)
      close(21)

!     bottom friction
      open (21,file='bldis.std',form='UNFORMATTED', &
      & access='DIRECT',recl=ilen*idim*2 )
      do i=1,idim
         read (Unit = 21,rec=i ) (blfr(j,i),j=1,idim)
      end do


!       ar1=(6370.0d0/700.0d0)**2
!       ar2=(6370.0d0/450.0d0)**2


!       coefficients for inversed PV 
          do i=1,idim
          di=evl(i)**2-evl(i)*(2*ar1+ar2)+ar1*ar2
          mi=evl(i)**2-2.0d0*evl(i)*(ar1+ar2)+3*ar1*ar2
          e0=evl(i)
          e1=evl(i)-ar1
          e2=evl(i)-ar2
          pinv(i,1)=1.0d0/e1+ar1**2*e2/(e0*e1*mi)
          pinv(i,2)=-ar1*e2/(e0*mi)
          pinv(i,3)=ar1*ar2/(e0*mi)
          pinv(i,4)=e1*e2/(e0*mi)
          pinv(i,5)=-ar2*e1/(e0*mi)
          pinv(i,6)=di/(e0*mi)
          end do



      tau2=tau/2.0d0
      tau1=tau

      call gettim(j1,j2,j3,j4)


      do istep=istep0,nstep

        do kstep=1,2


          do m=1,3

          do i=1,idim
          om2(i)=0.0d0
          om(i)=psi(i+(m-1)*idim)*evl(i)
          ps(i)=psi(i+(m-1)*idim)
          end do

          if (m.eq.1) then
          do i=1,idim
             om(i)=om(i)-ar1*(psi(i)-psi(i+idim))
          end do
          end if

          if (m.eq.2) then
          do i=1,idim
             om(i)=om(i)+ar1*(psi(i)-psi(i+idim)) &
     &                  -ar2*(psi(i+idim)-psi(i+2*idim))
          end do
          end if

          if (m.eq.3) then
          do i=1,idim
             om(i)=om(i)+ar2*(psi(i+idim)-psi(i+2*idim))
          end do
          end if

          do i=1,idim
             omp(i+(m-1)*idim)=om(i)
          end do

          if (m.eq.3) then
          do i=1,idim
             om(i)=om(i)+top(i)
          end do
          end if

          om(1)=om(1)+2.0d0/dsqrt(3.0d0)


          j=1
          do i=1,idim
            z1=0.
            z3=ps(i)
            z5=om(i)
          do while (fc(j).ne.0)
            a=fc(j)
            b=sc(j)
            zz=va(j)
            z1=z1 - zz *  ( ps(a) * om(b) - ps(b) * om(a) )
            om2(b)=om2(b) + zz * ( ps(a) * z5 - z3 * om(a)   )
            om2(a)=om2(a) - zz * ( ps(b) * z5 - z3 * om(b)   )
            j=j+1
          end do
          j=j+1
          om2(i)=z1+om2(i)

          end do

          do i=1,idim
          omm(i+(m-1)*idim)=om2(i)
          end do

         end do


          do i=1,idm
          buf(i)=0.
          end do 



          do i=1,idim
          buf(i)=alphaR*ar1*(psi(i) - psi(i+idim))
          buf(i+idim)=alphaR* ( - ar1*(psi(i) - psi(i+idim))  &
     &              + ar2*(psi(i+idim) - psi(i+2*idim)))
          buf(i+2*idim)=-alphaR*ar2*(psi(i+idim)-psi(i+2*idim))
          end do

          do m=1,3
          do i1=1,idim
          i=i1+(m-1)*idim
          buf(i)=buf(i)+ omm(i)
          buf(i)=buf(i)+ rpar*tem(i)    
          buf(i)=buf(i) - miuH*omp(i)*evl(i1)**4
          end do
          end do

          do i=1,idim
          z1=0.0d0
          do j=1,idim
         z1=z1+blfr(i,j)*psi(j+2*idim)
          end do
          buf(i+2*idim)=buf(i+2*idim)-z1*alphaE
          end do

          do i=1,idim
          f(i)=pinv(i,1)*buf(i)+pinv(i,2)*buf(i+idim)+&
     &          pinv(i,3)*buf(i+2*idim)
          f(i+idim)=pinv(i,2)*buf(i)+pinv(i,4)*buf(i+idim)+&
     &          pinv(i,5)*buf(i+2*idim)
          f(i+2*idim)=pinv(i,3)*buf(i)+pinv(i,5)*buf(i+idim)+&
     &          pinv(i,6)*buf(i+2*idim)
          end do


          do i=1,idm
            if (kstep.eq.1) then
              pso(i)=psi(i)
              psi(i)=f(i)*tau2+psi(i)
            else
              psi(i)=f(i)*tau1+pso(i)
           end if
          end do

        end do

        enerk=0.

        do  i=1,idim
            enerk=enerk-&
     &    evl(i)*(psi(i)**2+psi(i+idim)**2+psi(i+2*idim)**2)
        end do


        p=nout1*dble(istep)/dble(nout1)
        p1=nout1*int((istep)/(nout1))
        if(int(p).eq.int(p1)) then
        do i=1,idm
        psiout(i,nrc)=psi(i)
        end do
        nrc=nrc+1
        end if




        gotime=gotime+tau/pi
        p=nout*dble(istep)/dble(nout)
        p1=nout*int((istep)/(nout))
        if(int(p).eq.int(p1)) then
        call gettim(l1,l2,l3,l4)
        s3=(l1-j1)*60*60*100+   &
      &           (l2-j2)*60*100+(l3-j3)*100+l4-j4
        call gettim(j1,j2,j3,j4)

        write(*,'(A6,f15.6,A6,f12.8,a6,f9.6)') &
      &    ' DAY= ',real(gotime),' EKI= ',real(enerk), &
      &    ' Tim= ',real(s3/100)


      if (iout1.eq.1) then
      open (21,file='model.dt',form='UNFORMATTED',&
      & access='DIRECT',recl=idm*ilen*(nrc-1))
      write (Unit = 21,rec=nrc1 ) ((psiout(i,j),i=1,idm),j=1,nrc-1)
      close(21)
      end if

      if (iout2.eq.1) then
      open (21,file='modelg2.std',form='UNFORMATTED',&
      & access='DIRECT',recl=idg2*ilen/2)

        do jj=1,nrc-1
        do k0=1,idg2/2
        psig2(k0)=0.
        end do
        do j=1,idim
        do k0=1,idg2/2
        psig2(k0)=psig2(k0)+sp2gr2(k0+idg2/2,j)*psiout(j+idim,jj)
        end do
        end do
        write (Unit = 21,rec=nrc2) (psig2(i),i=1,idg2/2)
        nrc2=nrc2+1
        end do
        close(21)
      end if

      if (iout3.eq.1) then
!      open (21,file='modelg4.std',form='UNFORMATTED',&
!      & access='DIRECT',recl=idg4*ilen/2)

        do jj=1,nrc-1
        do mm=0,2
        do k0=1,idg4/2
        psig4(k0)=0.
        end do
        do j=1,idim
        do k0=1,idg4/2
        psig4(k0)=psig4(k0)+sp2gr4(k0+idg4/2,j)*psiout(j+mm*idim,jj)
        end do
        end do
!        write (Unit = 21,rec=nrc3) (psig4(i),i=1,idg4/2)
        write (20,'(3000f10.6)') (psig4(i),i=1,idg4/2)
        nrc3=nrc3+1
        end do
        end do
        close(21)
      end if


        nrc1=nrc1+1

        open(25,file='init')
        do i=1,idm
          write(25,*) psi(i)
        end do
        write(25,*) gotime
        close(25)

         nrc=1
       end if
       end do

       end



      subroutine eform(evl)
      Implicit Real*8 (a-h, o-z)
      parameter ( idim=171, iln=37064, iaz=19, ilen=1 )
      double precision evl(idim)
      integer nrc,m,n,i,k
      nrc=1
      do m=1,iaz
         do n=m,iaz
           if ( (((m+n)/2)*2).ne.(m+n) ) then
           zn=dble(n)
           if ( (m.ne.1)) then
              evl(nrc)=-zn*(zn-1.0d0)
              evl(nrc+1)=-zn*(zn-1.0d0)
                nrc=nrc+2
             else
               if (n.ne.1) then
               evl(nrc)=-zn*(zn-1.0d0)
               nrc=nrc+1
               end if
            end if
            end if
           end do
        end do

        return
        end


