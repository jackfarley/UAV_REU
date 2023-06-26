

figure('Position', [200 200 600 600 ]);
FONTSIZE_NUMBERS = 10;
COLOR_NUMBERS_ONE = 'r';
COLOR_NUMBERS_TWO = 'b';
CIRCLE_RADIUS = 0.5;
hold on
grid on
xlabel('X coordinate', 'FontSize',14)
ylabel('Y coordinate','FontSize',14)
circle(20,20,0)
circle(0,0,0)
line([20 20],[20 0],'Color','black','LineStyle','-','LineWidth',2);
line([0 20],[20 20],'Color','black','LineStyle','-','LineWidth',2);
line([0 0],[20 0],'Color','black','LineStyle','-','LineWidth',2);
line([0 20],[0 0],'Color','black','LineStyle','-','LineWidth',2);
xticks(0:1:20)
yticks(0:1:20)
text(4,4,'o','FontSize',10)
text(4,3.8,'start','FontSize',10)
text(2,12,'o','FontSize',10)
text(2,11.8,'end','FontSize',10)
circle(2,6,2)
circle(11,10,2)
circle(18,11,2)
circle(10,18,2)



[(2, 1)]
[(3, 1)]
[(4, 1)]
[(4, 2)]
[(5, 2)]
[(6, 2)]
[(7, 2)]
[(8, 2)]
[(9, 2)]
[(10, 2)]
[(11, 2)]
[(12, 2)]
[(13, 2)]
[(14, 2)]
[(15, 2)]
[(16, 2)]S

X_LOCATIONS_1 = [4.0 6.0 8.0 10.0 12.0 15.0 16.0 16.0 16.0 16.0 18.0 16.0 15.0 12.0 9.0 6.0 9.0 10.0 10.0 11.0 12.0 10.0 7.0 4.0 2.0 ];
Y_LOCATIONS_1 = [4.0 6.0 8.0 10.0 12.0 14.0 11.0 11.0 11.0 11.0 8.0 11.0 12.0 14.0 16.0 18.0 17.0 14.0 14.0 11.0 8.0 9.0 10.0 11.0 12.0 ];
movingPoint_one = rectangle('Parent',gca,'Position',[0,0,0.5,0.5],'Curvature',[1,1],'FaceColor','b');

for frame=1:25
set(movingPoint_one,'Position',[X_LOCATIONS_1(frame),Y_LOCATIONS(frame),.5,.5])
frames(frame)=getframe;
pause(1)
end
