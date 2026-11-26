const menuToggle = document.getElementById('menu-toggle');
const navLinks = document.getElementById('nav-links');

menuToggle.addEventListener('click', function(){
    navLinks.classList.toggle('active');
});









const track = document.getElementById("carousel-track");
const slides = document.querySelectorAll(".slide");

let i = 0;
function nextSlide(){

    i++;
    if(i>= slides.length){
        i= 0;
    }

    track.style.transform = `translateX(-${i*100}%)`;
}


setInterval(nextSlide, 5000);













const form = document.getElementById('registerForm');

if(form){

form.addEventListener('submit', function(e){

    e.preventDefault();

    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const age = document.getElementById('age').value;
    const troop = document.getElementById('troop').value;
    const reason = document.getElementById('reason').value;

    const male = document.getElementById("male");
    const female = document.getElementById("female");

    let gender = "";

    if(male.checked){
        gender = "Male";
    }
    else if(female.checked){
        gender = "Female";
    }


    if(name.length < 3){
        alert('Please input a valid name (>2 characters)');
        return;
    }


    if(!email.includes('@')){
        alert('Please input a valid email address');
        return;
    }


    if(gender === ""){
        alert("Please choose gender");
        return;
    }


    if(age < 16 || age > 150){
        alert('Age must be older than 16 years old');
        return;
    }
    if(troop === ''){
        alert('Please choose your troop');
        return;
    }
    if(reason.length < 5){
        alert('Please provide longer text for your reason to join (>4 characters)');
        return;
    }


    alert(`Welcome ${name}, you are now a Clasher!`);

});

}









