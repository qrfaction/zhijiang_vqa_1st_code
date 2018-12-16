



data_path = '../data/'

te_video_path = data_path + 'test/'

tr_video_path = data_path + 'train/'

TEXT_LEN = 14

model_path = './model_dir/'

word_vec_dir = '../data/word_vec/'

output_dir = '../submit/'

ATTR_LEN = 2

cfg = {
    'alpha':0.65
}


fix_error = {
    "ZJL2557,what is in the man's hands in the video,basketball,basketball basketball,what is color of trousers of man in video,red,dark-red,red,what is man doing in video,talking,explaining,doing program,where is man in video,basketball park,gymnasium,basketball stadium,is man standing or sitting in video,,standing,standing,standing":
    "ZJL2557,what is in the man's hands in the video,basketball,basketball,basketball,what is color of trousers of man in video,red,dark-red,red,what is man doing in video,talking,explaining,doing program,where is man in video,basketball park,gymnasium,basketball stadium,is man standing or sitting in video,standing,standing,standing",

    "ZJL730,is the girl's hair in the video long hair or short hair,hair,hair,hair,what color clothes is the girl wearing in the video,red,rose red,purple red,what is the girl doing in the video,speaking,speaking,speaking,where is the girl in the video,on lawn,on lawn,on lawn,is the girl standing or sitting in the video,standing,standing,standing":
    "ZJL730,is the girl's hair in the video long hair or short hair,long hair,long hair,long hair,what color clothes is the girl wearing in the video,red,rose red,purple red,what is the girl doing in the video,speaking,speaking,speaking,where is the girl in the video,on lawn,on lawn,on lawn,is the girl standing or sitting in the video,standing,standing,standing",

    "ZJL3127,what is at the back of the door,toilet,trash bin,frame,what is the color of the clothes the man wears in the video,blue,dark blue,black,blue,what is man doing in video,taking rest,reading book,throwing book,where is man in video,indoor,in washing room,in the house,was man sitting or lying at beginning in video,lying,lying lying":
    "ZJL3127,what is at the back of the door,toilet,trash bin,frame,what is the color of the clothes the man wears in the video,blue,dark blue,blue,what is man doing in video,taking rest,reading book,throwing book,where is man in video,indoor,in washing room,in the house,was man sitting or lying at beginning in video,lying,lying,lying",

    "ZJL448,what is the woman holding in the video,cup,cup,cup,what color is the wall in the video,white,white,white,what is the woman in the video doing,cleaning,cleaning,cleaning,the table,where is sofa in video,against wall,against wall,against the wall,is woman in video standing or sitting standing,standing,standing":
    "ZJL448,what is the woman holding in the video,cup,cup,cup,what color is the wall in the video,white,white,white,what is the woman in the video doing,cleaning,cleaning,cleaning,where is sofa in video,against wall,against wall,against the wall,is woman in video standing or sitting,standing,standing,standing",

    "ZJL2531,what is on the wooden table,fruits,pomegranate,plate,,what is color of man's hair in cartoon,yellow,golden-yellow,golden,what is man doing in cartoon,talking,explaining,doing program,where is packaging milk,on cupboard,on closet next to bread,is the man sitting or standing in the cartoon,standing,standing,standing":
    "ZJL2531,what is on the wooden table,fruits,pomegranate,plate,what is color of man's hair in cartoon,yellow,golden-yellow,golden,what is man doing in cartoon,talking,explaining,doing program,where is packaging milk,on cupboard,on closet,next to bread,is the man sitting or standing in the cartoon,standing,standing,standing",

    "ZJL1774,what is on the road in video,car,car,car,what is the color of the car in the video,blue,navy blue,dark,blue,what is person in video doing,driving,driving,driving,is headlight in video open or closed,closed,closed,closedis the person in the car is standing or sitting in the video,sitting,sitting,sitting":
    "ZJL1774,what is on the road in video,car,car,car,what is the color of the car in the video,blue,navy blue,blue,what is person in video doing,driving,driving,driving,is headlight in video open or closed,closed,closed,closed,is the person in the car is standing or sitting in the video,sitting,sitting,sitting",

    "ZJL458,what is the woman holding in the video,cup,cup,cup,what color is the wall in the video,white,white,white,what is the woman in the video doing,cleaning,cleaning,cleaning,the table,where is sofa in video,against wall,against wall,against the wall,is woman in video standing or sitting standing,standing,standing":
    "ZJL458,what is the woman holding in the video,cup,cup,cup,what color is the wall in the video,white,white,white,what is the woman in the video doing,cleaning,cleaning,cleaning,where is sofa in video,against wall,against wall,against the wall,is woman in video standing or sitting,standing,standing,standing",

    "ZJL11421,what is in the video,people,plate,gas cooker,where are people,kitchen,indoor,near countertop,what color of pants does the man wear,pink,pale pink,light pink,what is the woman doing,speaking,wiping plate,turning around,does the woman have long hair or short hair,standing,standing,standing":
    "ZJL11421,what is in the video,people,plate,gas cooker,where are people,kitchen,indoor,near countertop,what color of pants does the man wear,pink,pale pink,light pink,what is the woman doing,speaking,wiping plate,turning around,does the woman have long hair or short hair,long hair,long hair,long hair",

    "ZJL2551,what is on the stand,audience,song fans,fans,what is the woman in dress doing,singing,singing,singing,what is the color of the singing woman's hair,black,black,black,what is the singing woman holding,microphone,voicetube,wireless-voice tube,is the singing woman or sitting in the video,standing,standing,standing":
    "ZJL2551,what is on the stand,audience,song fans,fans,what is the woman in dress doing,singing,singing,singing,what is the color of the singing woman's hair,black,black,black,what is the singing woman holding,microphone,voicetube,wireless-voice tube,is the singing woman standing or sitting in the video,standing,standing,standing",

    "ZJL9791,what is in video,window,cabinet,television,where are the people in video,in front of window,next to christmas tree,indoor,what color is the wall in video,white,milk white,off-white,what are the people doing in video,taking things,hanging things,smiling,does the woman in video wear long sleeves or short sleeves,closed,closed,closed":
    "ZJL9791,what is in video,window,cabinet,television,where are the people in video,in front of window,next to christmas tree,indoor,what color is the wall in video,white,milk white,off-white,what are the people doing in video,taking things,hanging things,smiling,does the woman in video wear long sleeves or short sleeves,long sleeves,long sleeves,long sleeves",

    "ZJL837,what is in the video,drum set,microphone,guitar,what color pants does the last person wear,white,white,white,what is the person doing in the video,singing,paying guitar,driving car,where is the person singing in the video,indoor,house,living room,is the last person in the video sitting or standing,standing,standing,standing":
    "ZJL837,what is in the video,drum set,microphone,guitar,what color pants does the last person wear,white,white,white,what is the person doing in the video,singing,playing guitar,driving car,where is the person singing in the video,indoor,house,living room,is the last person in the video sitting or standing,standing,standing,standing",

    "ZJL11725,what is in the video,people,guitar,microphone,where are people in the video,where is seat in video,on train,in car,on bus,what is color of chair in car,blue,light blue,dark blue,what are people doing in video,having party,singing,cheering,what musical instruments are in video,guitar,drum":
    "ZJL11725,what is in the video,people,guitar,microphone,where are people in the video,on train,in car,on bus,what is color of chair in car,blue,light blue,dark blue,what are people doing in video,having party,singing,cheering,what musical instruments are in video,guitar,drum,guitar",

    "ZJL3517,what is on the face of the person in the video,mustache,mustache,mustache,what color t-shirt does the man wear in the video,red,peach,pink,light red,what is person doing in video,unlacing shoelace,unlacing shoelace unlacing shoelace,where is the person in the video,in bathroom,in bathroom,in bathroom,is the person in the video standing or sitting,sitting,sitting,sitting":
    "ZJL3517,what is on the face of the person in the video,mustache,mustache,mustache,what color t-shirt does the man wear in the video,red,peach,light red,what is person doing in video,unlacing shoelace,unlacing shoelace,unlacing shoelace,where is the person in the video,in bathroom,in bathroom,in bathroom,is the person in the video standing or sitting,sitting,sitting,sitting",

    "ZJL1582,what is in the hand of the person who wears a horse mask,paper,paper,paper,what color are the clothes of the person who wears a horse mask in the video,red,red,red,what is the woman doing,scolding person who wears horse mask,scolding person who wears horse mask,scolding person who wears horse mask,does the person who wears a horse mask break the glass,yes,yes,yes,is the person who wears a horse maskin the video standing or sitting,standing,standing,standing":
    "ZJL1582,what is in the hand of the person who wears a horse mask,paper,paper,paper,what color are the clothes of the person who wears a horse mask in the video,red,red,red,what is the woman doing,scolding person who wears horse mask,scolding person who wears horse mask,scolding person who wears horse mask,does the person who wears a horse mask break the glass,yes,yes,yes,is the person who wears a horse mask in the video standing or sitting,standing,standing,standing",

    "ZJL8928,what is in the video,people,sofa,table,where is the person in the video,on sofa,indoors,living room,what color is the person in the video wearing,black,gray black,dark black,what is the person doing in the video,standing up,walking,taking plastic bag,whether is it day or nightin the video,day,day,day":
    "ZJL8928,what is in the video,people,sofa,table,where is the person in the video,on sofa,indoors,living room,what color is the person in the video wearing,black,gray black,dark black,what is the person doing in the video,standing up,walking,taking plastic bag,whether is it day or night in the video,day,day,day",

    "ZJL3600,what is hanging next to the mirror in video,painting,painting,painting,what is the color of the person's clothes in video,grey,grey black,sliver grey,is the person sitting or standing in the video,standing,standing,standing,what is the person doing in video,putting on clothes,looking at mirror,taking something,what is the color of the wall in video,blue,skyblue,light blue":
    "ZJL3600,what is hanging next to the mirror in video,painting,painting,painting,what is the color of the person's clothes in video,grey,grey black,sliver grey,is the person sitting or standing in the video,standing,standing,standing,what is the person doing in video,putting on clothes,looking at mirror,taking something,what is the color of the wall in video,blue,sky blue,light blue",

    "ZJL3589,where is the person in video,near window,near window,near window,what is the color of the person's clothes in video,grey,light grey,grey white,what is the person doing in video,eating,eating,eating,is the person sitting or standing in the video,standing,standing,standing,what is the color of the wall in video,yellow,pale brown,lightyellow":
    "ZJL3589,where is the person in video,near window,near window,near window,what is the color of the person's clothes in video,grey,light grey,grey white,what is the person doing in video,eating,eating,eating,is the person sitting or standing in the video,standing,standing,standing,what is the color of the wall in video,yellow,pale brown,light yellow",

    "ZJL10008,what is in the video,person,chair,instrument,where are the people in the video,indoor,on chair,in front of score,what color hair does the person have in the video,black,pure black,dark black,what is the person in the video doing,paying instrument,moving fingers,looking at score,is there one person or multiple people in the video,one,one,one":
    "ZJL10008,what is in the video,person,chair,instrument,where are the people in the video,indoor,on chair,in front of score,what color hair does the person have in the video,black,pure black,dark black,what is the person in the video doing,playing instrument,moving fingers,looking at score,is there one person or multiple people in the video,one,one,one",

    "ZJL9317,what is in the video,plate,paper box,bottle,where is the paper box,on floor,near wall,warehouse,what color of the clothes does the person in the video wear,black,aterrimus,furvous,what is the person in the video doing,taking bottle,sneezing,eating,is the person in the video squatting of sitting at the end,squatting,squatting,squatting":
    "ZJL9317,what is in the video,plate,paper box,bottle,where is the paper box,on floor,near wall,warehouse,what color of the clothes does the person in the video wear,black,aterrimus,furvous,what is the person in the video doing,taking bottle,sneezing,eating,is the person in the video squatting or sitting at the end,squatting,squatting,squatting",

    "ZJL9246,what is in the video,refrigerator,trash bin,hearth,where is the trash bin,on floor,beside wall,kitchen,what color of the clothes does the person in the video wear,purple,dark purple,dull purple,what is the person in the video doing,drinking water,turning on cooking fire,mixing,is the door of the refrigerator openor closed,closed,closing,locked":
    "ZJL9246,what is in the video,refrigerator,trash bin,hearth,where is the trash bin,on floor,beside wall,kitchen,what color of the clothes does the person in the video wear,purple,dark purple,dull purple,what is the person in the video doing,drinking water,turning on cooking fire,mixing,is the door of the refrigerator open or closed,closed,closing,locked",

    "ZJL1161,what is the gender of the person in the video,male,male,male,what color is the belt in the video,black,black,black,what is on the head in the video,hat,hat,hat,what is the first person in the video doing,speaking,speaking,speaking,is the person taking the box in the video sitting still standing,standing,standing,standing":
    "ZJL1161,what is the gender of the person in the video,male,male,male,what color is the belt in the video,black,black,black,what is on the head in the video,hat,hat,hat,what is the first person in the video doing,speaking,speaking,speaking,is the person taking the box in the video sitting or standing,standing,standing,standing",

    "ZJL2793,what color clothes does person in the video wear,gray,gray white,blue-gray,what is the person in the video doing,taking water glass,drinking water,watching tv,is the person in the video standing orsquatting,squatting,squatting,squatting,where is the person in the video,living room,living room,living room,how many tvs are there on the table in the video,two,two,two":
    "ZJL2793,what color clothes does person in the video wear,gray,gray white,blue-gray,what is the person in the video doing,taking water glass,drinking water,watching tv,is the person in the video standing or squatting,squatting,squatting,squatting,where is the person in the video,living room,living room,living room,how many tvs are there on the table in the video,two,two,two",

    "ZJL1153,what is in the video,table,chair,arrow,what color are the people wearing glasses in the video,pink,pale pink,pink,what is the person wearing glasses doing in the video,reading newspaper,reading newspaper,reading newspaper,where is the press in the video at the beginning,in hand,in hand,in hand,is the person shot by an arrow sitting and standing in the video,sitting,sitting,sitting":
    "ZJL1153,what is in the video,table,chair,arrow,what color are the people wearing glasses in the video,pink,pale pink,pink,what is the person wearing glasses doing in the video,reading newspaper,reading newspaper,reading newspaper,where is the press in the video at the beginning,in hand,in hand,in hand,is the person shot by an arrow sitting or standing in the video,sitting,sitting,sitting",

    "ZJL1720,what is on the table in the video,cup,cup,cup,how many pictures are there in the video,three,three,three,what color is the man is clothes in the first picture,white,white,white,what are the two people in the second picture of the video doing,hug,hug,hug,is the person in the first picture standing sitting in the video,standing,standing,standing":
    "ZJL1720,what is on the table in the video,cup,cup,cup,how many pictures are there in the video,three,three,three,what color is the man is clothes in the first picture,white,white,white,what are the two people in the second picture of the video doing,hug,hug,hug,is the person in the first picture standing or sitting in the video,standing,standing,standing",

}


