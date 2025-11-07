package com.leareng.service;

import com.leareng.entity.User;
import com.leareng.entity.VerificationToken;
import com.leareng.repository.UserRepository;
import com.leareng.repository.VerificationTokenRepository;
import com.leareng.security.JwtUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;

@Service
public class UserService {
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private VerificationTokenRepository tokenRepository;
    
    @Autowired
    private PasswordEncoder passwordEncoder;
    
    @Autowired
    private JwtUtil jwtUtil;
    
    @Autowired
    private EmailService emailService;
    
    public User registerUser(String email, String password) {
        if (userRepository.existsByEmail(email)) {
            throw new RuntimeException("이미 등록된 이메일입니다.");
        }
        
        User user = new User();
        user.setEmail(email);
        user.setPassword(passwordEncoder.encode(password));
        user.setIsVerified(false);
        user.setIsAdmin(false);
        user = userRepository.save(user);
        
        // Generate verification token
        String token = jwtUtil.generateToken(email);
        VerificationToken verificationToken = new VerificationToken();
        verificationToken.setEmail(email);
        verificationToken.setToken(token);
        verificationToken.setExpiresAt(LocalDateTime.now().plusHours(24));
        tokenRepository.save(verificationToken);
        
        // Send verification email
        emailService.sendVerificationEmail(email, token);
        
        return user;
    }
    
    public String login(String email, String password) {
        Optional<User> userOpt = userRepository.findByEmail(email);
        if (userOpt.isEmpty() || !passwordEncoder.matches(password, userOpt.get().getPassword())) {
            throw new RuntimeException("이메일 또는 비밀번호가 올바르지 않습니다.");
        }
        
        User user = userOpt.get();
        if (!user.getIsVerified() && !user.getIsAdmin()) {
            throw new RuntimeException("이메일 인증이 필요합니다. 이메일을 확인해주세요.");
        }
        
        return jwtUtil.generateToken(email);
    }
    
    @Transactional
    public boolean verifyEmail(String email, String token) {
        Optional<VerificationToken> tokenOpt = tokenRepository.findByToken(token);
        if (tokenOpt.isEmpty()) {
            return false;
        }
        
        VerificationToken verificationToken = tokenOpt.get();
        if (!verificationToken.getEmail().equals(email) || 
            verificationToken.getExpiresAt().isBefore(LocalDateTime.now())) {
            return false;
        }
        
        Optional<User> userOpt = userRepository.findByEmail(email);
        if (userOpt.isEmpty()) {
            return false;
        }
        
        User user = userOpt.get();
        user.setIsVerified(true);
        userRepository.save(user);
        
        tokenRepository.delete(verificationToken);
        return true;
    }
    
    public void requestPasswordReset(String email) {
        Optional<User> userOpt = userRepository.findByEmail(email);
        if (userOpt.isEmpty()) {
            throw new RuntimeException("등록되지 않은 이메일입니다.");
        }
        
        // Delete old tokens
        tokenRepository.deleteByEmail(email);
        
        // Generate new token
        String token = jwtUtil.generateToken(email);
        VerificationToken verificationToken = new VerificationToken();
        verificationToken.setEmail(email);
        verificationToken.setToken(token);
        verificationToken.setExpiresAt(LocalDateTime.now().plusHours(24));
        tokenRepository.save(verificationToken);
        
        emailService.sendPasswordResetEmail(email, token);
    }
    
    @Transactional
    public boolean resetPassword(String token, String newPassword) {
        Optional<VerificationToken> tokenOpt = tokenRepository.findByToken(token);
        if (tokenOpt.isEmpty()) {
            return false;
        }
        
        VerificationToken verificationToken = tokenOpt.get();
        if (verificationToken.getExpiresAt().isBefore(LocalDateTime.now())) {
            return false;
        }
        
        Optional<User> userOpt = userRepository.findByEmail(verificationToken.getEmail());
        if (userOpt.isEmpty()) {
            return false;
        }
        
        User user = userOpt.get();
        user.setPassword(passwordEncoder.encode(newPassword));
        userRepository.save(user);
        
        tokenRepository.delete(verificationToken);
        return true;
    }
    
    public Optional<User> findByEmail(String email) {
        return userRepository.findByEmail(email);
    }
    
    public void resendVerificationEmail(String email) {
        Optional<User> userOpt = userRepository.findByEmail(email);
        if (userOpt.isEmpty()) {
            throw new RuntimeException("등록되지 않은 이메일입니다.");
        }
        
        User user = userOpt.get();
        if (user.getIsVerified()) {
            throw new RuntimeException("이미 인증된 계정입니다.");
        }
        
        // Delete old tokens
        tokenRepository.deleteByEmail(email);
        
        // Generate new token
        String token = jwtUtil.generateToken(email);
        VerificationToken verificationToken = new VerificationToken();
        verificationToken.setEmail(email);
        verificationToken.setToken(token);
        verificationToken.setExpiresAt(LocalDateTime.now().plusHours(24));
        tokenRepository.save(verificationToken);
        
        // Resend verification email
        emailService.sendVerificationEmail(email, token);
    }
}

