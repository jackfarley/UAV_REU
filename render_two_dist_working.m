

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



[(2, 1), (19, 20)]
[(3, 1), (18, 20)]
[(4, 1), (17, 20)]
[(5, 1), (16, 20)]
[(6, 1), (15, 20)]
[(7, 1), (14, 20)]
[(8, 1), (13, 20)]
[(9, 1), (12, 20)]
[(10, 1), (11, 20)]
[(11, 1), (10, 20)]
[(12, 1), (9, 20)]
[(13, 1), (8, 20)]
[(14, 1), (7, 20)]
[(15, 1), (6, 20)]
[(16, 1), (5, 20)]


both just started in corners and one goes rights, the other goes left


X_LOCATIONS_1 = [4.0 6.0 8.0 10.0 12.0 15.0 16.0 16.0 16.0 16.0 18.0 16.0 15.0 12.0 9.0 6.0 9.0 10.0 10.0 11.0 12.0 10.0 7.0 4.0 2.0 ];
Y_LOCATIONS_1 = [4.0 6.0 8.0 10.0 12.0 14.0 11.0 11.0 11.0 11.0 8.0 11.0 12.0 14.0 16.0 18.0 17.0 14.0 14.0 11.0 8.0 9.0 10.0 11.0 12.0 ];
movingPoint_one = rectangle('Parent',gca,'Position',[0,0,0.5,0.5],'Curvature',[1,1],'FaceColor','b');

for frame=1:25
set(movingPoint_one,'Position',[X_LOCATIONS_1(frame),Y_LOCATIONS(frame),.5,.5])
frames(frame)=getframe;
pause(1)
end
