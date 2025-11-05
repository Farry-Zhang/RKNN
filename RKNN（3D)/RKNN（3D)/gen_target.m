function truth= gen_target(model)

%variables
truth.K= 200;                   %length of data/number of scans
truth.X= cell(truth.K,1);             %ground truth for states of targets
truth.N= zeros(truth.K,1);            %ground truth for number of targets
truth.L= cell(truth.K,1);             %ground truth for labels of targets (k,i)
truth.track_list= cell(truth.K,1);    %absolute index target identities (plotting)
truth.total_tracks= 0;          %total number of appearing tracks

%target initial states and birth/death times
nbirths= 15;

xstart(:,1)  = [ 80000; -250; 20000;0; 4500; 0];        tbirth(1)  = 1;     tdeath(1)  = truth.K;
xstart(:,2)  = [ 80000; -250; 10000;0; 4500; 0];        tbirth(2)  = 1;     tdeath(2)  = 100;
xstart(:,3)  = [ 80000; -250; -10000;0; 4500; 0];       tbirth(3)  = 1;     tdeath(3)  = truth.K;
xstart(:,4)  = [ 80000; -250; -20000;0; 4500; 0];       tbirth(4)  = 1;     tdeath(4)  = 100;

xstart(:,5)  = [ 75000; -220; 30000;0; 4500; 0];        tbirth(5)  = 10;     tdeath(5)  = truth.K;
xstart(:,6)  = [ 75000; -220; -30000;0; 4500; 0];        tbirth(6)  = 10;     tdeath(6)  = 120;

xstart(:,7)  = [ 70000; -210; 40000;0; 4500; 0];        tbirth(7)  = 20;     tdeath(7)  = 120;
xstart(:,8)  = [ 70000; -210; 25000;0; 4500; 0];        tbirth(8)  = 20;     tdeath(8)  = 120;
xstart(:,9)  = [ 70000; -210; -25000;0; 4500; 0];        tbirth(9)  = 20;     tdeath(9)  = truth.K;
xstart(:,10)  = [ 70000; -210; -40000;0; 4500; 0];        tbirth(10)  = 20;     tdeath(10)  = truth.K;

xstart(:,11)  = [ 65000; -200; 35000;0; 4500; 0];        tbirth(11)  = 30;     tdeath(11)  = truth.K;
xstart(:,12)  = [ 65000; -200; 15000;0; 4500; 0];        tbirth(12)  = 30;     tdeath(12)  = 150;
xstart(:,13)  = [ 65000; -200; 0;0; 4500; 0];            tbirth(13)  = 30;     tdeath(13)  = truth.K;
xstart(:,14)  = [ 65000; -200; -15000;0; 4500; 0];        tbirth(14)  = 30;     tdeath(14)  = 150;
xstart(:,15)  = [ 65000; -200; -35000;0; 4500; 0];        tbirth(15)  = 30;     tdeath(15)  = truth.K;

%generate the tracks
for targetnum=1:nbirths
    targetstate = xstart(:,targetnum);
    for k=tbirth(targetnum):min(tdeath(targetnum),truth.K)
        truth.X{k}= [truth.X{k} targetstate];
        targetstate = gen_newstate_target(model,targetstate,'noise');
        truth.track_list{k} = [truth.track_list{k} targetnum];
        truth.N(k) = truth.N(k) + 1;
    end    
end
truth.total_tracks= nbirths;



