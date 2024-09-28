const body = document.querySelector('body'),
      sidebar = body.querySelector('.sidebar'),
      toggle = body.querySelector('.toggle')
      searchBtn = body.querySelector('.search-box'),
      modeSwitch = body.querySelector('.toggle-switch'),
      modeText = body.querySelector('.mode-text');

const navLinks = document.getElementsByClassName('nav-link')

;[...navLinks].forEach(navLink => {
    navLink.addEventListener('click', isActive)
});

function isActive(){
    const current = this
    const activeLink = body.querySelector('.nav-link.active')
    activeLink.classList.remove('active')
    current.classList.add('active')
}

toggle.addEventListener('click', () => {
    sidebar.classList.toggle('close')
})

modeSwitch.addEventListener('click', () => {
    body.classList.toggle('dark')

    if(body.classList.contains('dark')){
        modeText.innerText = "Light Mode"
    }else{
        modeText.innerText = "Dark Mode"
    }
})