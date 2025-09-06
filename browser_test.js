console.log("Testing horizontal steps UI...");

// Quick browser test - paste this in browser console
setTimeout(() => {
  const horizontalContainer = document.querySelector('.horizontal-steps-container');
  const oldLiveProgress = document.querySelector('.live-progress');
  
  console.log('Horizontal container found:', !!horizontalContainer);
  console.log('Old live progress found:', !!oldLiveProgress);
  
  if (horizontalContainer) {
    console.log('✅ Horizontal steps container is present!');
    const timeline = horizontalContainer.querySelector('.horizontal-steps-timeline');
    const circles = horizontalContainer.querySelectorAll('.horizontal-step-circle');
    console.log('Timeline found:', !!timeline);
    console.log('Step circles found:', circles.length);
    
    // Check CSS
    const computedStyle = getComputedStyle(timeline);
    console.log('Timeline display:', computedStyle.display);
    console.log('Timeline flex-direction:', computedStyle.flexDirection);
  } else {
    console.log('❌ Horizontal container not found - old component might still be rendering');
  }
  
  if (oldLiveProgress) {
    console.log('⚠️ Old live-progress component still present!');
  }
}, 2000);
