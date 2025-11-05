function handles= plot_truth(truth)

[X_track,k_birth,k_death]= extract_tracks(truth.X,truth.track_list,truth.total_tracks);
%plot ground truths
figure; truths= gcf; hold on;
set(gca,'FontSize',18);
set(gca,'GridLineStyle','-.','GridColor','k','GridAlpha',0.6)
set(gcf,'unit','centimeters','position',[15 2 18 15])
grid on
zlim([0 10])
xlabel('X [km]')
ylabel('Y [km]')
zlabel('Z [km]')
for i=1:truth.total_tracks
    Pt= X_track(:,k_birth(i):1:k_death(i),i); Pt=Pt([1 3 5],:)./1000;
    hlined1=plot3(Pt(1,:),Pt(2,:),Pt(3,:),'LineStyle','-','Marker','none','LineWidth',1,'Color',0*ones(1,3));
    %plot( Pt(1,:),Pt(2,:),'k-'); 
    hold on
    Hd1=plot3( Pt(1,1), Pt(2,1),Pt(3,1), 'ko','MarkerSize',6);
    % text(Pt(1,1)+1, Pt(2,1)+1,Pt(3,1),['T' num2str(i)]);
    hold on
    Hd2=plot3( Pt(1,(k_death(i)-k_birth(i)+1)), Pt(2,(k_death(i)-k_birth(i)+1)),Pt(3,(k_death(i)-k_birth(i)+1)), 'k^','MarkerSize',6);
    hold on
end
view(50,50)
hold on
%plot tracks and measurments in x/y
%figure; clf; 
tracking= gcf;

legend([Hd1,Hd2,hlined1],'航迹起始','航迹终止','目标航迹');
end

function [X_track,k_birth,k_death]= extract_tracks(X,track_list,total_tracks)

K= size(X,1); 
x_dim= size(X{K},1); 
k=K-1; while x_dim==0, x_dim= size(X{k},1); k= k-1; end
X_track= NaN(x_dim,K,total_tracks);
k_birth= zeros(total_tracks,1);
k_death= zeros(total_tracks,1);

max_idx= 0;
for k=1:K
    if ~isempty(X{k})
        X_track(:,k,track_list{k})= X{k};
    end
    if max(track_list{k})> max_idx %new target born?
        idx= find(track_list{k}> max_idx);
        k_birth(track_list{k}(idx))= k;
    end
    if ~isempty(track_list{k}), max_idx= max(track_list{k}); end
    k_death(track_list{k})= k;
end
end