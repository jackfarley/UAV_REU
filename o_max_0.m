

figure('Position', [200 200 600 600 ]);
FONTSIZE_NUMBERS = 10;
COLOR_NUMBERS_ONE = 'r';
COLOR_NUMBERS_TWO = 'b';
CIRCLE_RADIUS = 0.5;
hold on
grid on
xlabel('o_{max} = 1', 'FontSize',14)
ylabel('Y coordinate','FontSize',14)
circle(20,20,0)
circle(0,0,0)
line([20 20],[20 0],'Color','black','LineStyle','-','LineWidth',2);
line([0 20],[20 20],'Color','black','LineStyle','-','LineWidth',2);
line([0 0],[20 0],'Color','black','LineStyle','-','LineWidth',2);
line([0 20],[0 0],'Color','black','LineStyle','-','LineWidth',2);
xticks(0:1:20)
yticks(0:1:20)


%#GBS were at 1,1, 20,1, 5,5, 10,10, and 15,5










x = [1,1; 2, 1; 3, 1; 4, 1;4, 2; 5, 2;6, 2;6,3;7,3 ; 7,4 ;7,5 ; 8,5; 8,6; 8,7; 9,7; 10,7; 11,7 ; 12, 7; 13,7; 14,7 ;14,6; 14,5; 14,4 ;14,3; 15,3; 16,3; 16,2; 17,2; 18,2; 18,1; 19,1; 20,1 ];


text(1,1,'start',FontSize=10)
text(20,1, 'end', FontSize=10)
circle(1,1,3.5)
circle(5,5,3.5)
circle(10,10, 3.5)
circle(15,5,3.5)
circle(20,1, 3.5)


movingPoint_one = rectangle('Parent',gca,'Position',[0,0,0.5,0.5],'Curvature',[1,1],'FaceColor','b');

for frame=1:length(x)
set(movingPoint_one,'Position',[x(frame,1),x(frame,2),.75,.75])
frames(frame)=getframe;
pause(.4)
end
